"""Attribute MoE primitives for spatial x variable routing.

One primitive bank with a shared center. Each primitive i holds G expert slots
and routing logits r[i, v, :] that choose which slot variable v uses at
primitive i. Which attributes live in the slots is configurable
(expert_attrs, a subset of {scale, rot, intensity}); the others are shared by
all slots of the primitive:

    scale       slot g has its own scale       (else one scale per primitive)
    rot         slot g has its own rotation    (else one rotation per primitive)
    intensity   slot g has its own intensity b[i, g], shared by every variable
                routed to it (else each variable keeps its own a[i, v])

    f_v(x) = sum_i sum_g softmax(r[i, v] / tau)_g * I[i, v, g] * Gauss(x; xyz_i, scale_{i, g}, rot_{i, g})

with I[i, v, g] = b[i, g] for intensity experts and a[i, v] otherwise. When
routing is hard, each (i, v) uses only its argmax slot.

The parent class's pruning and densification helpers work on the envelope
shape (the largest-volume scale slot of each primitive), which is all
count_query, pruning and split/clone decisions need.
"""
import math

import torch
from torch import nn

from scene.gaussian_model import GaussianModel
from utils.general_utils import build_rotation

CENTER_FLOATS = 3
SCALE_FLOATS = 3
ROT_FLOATS = 4
INTENSITY_MIN = 1.0 / 255.0
EXPERT_ATTRS = ("scale", "rot", "intensity")


class AttrMoEGaussianModel(GaussianModel):
    def __init__(self, x_resol, y_resol, z_resol, num_variables, num_shapes, expert_attrs=("scale", "rot")):
        unknown = set(expert_attrs) - set(EXPERT_ATTRS)
        if unknown or not expert_attrs:
            raise ValueError(f"expert_attrs must be a non-empty subset of {EXPERT_ATTRS}, got {expert_attrs}")
        G = num_shapes
        self.expert_attrs = tuple(a for a in EXPERT_ATTRS if a in expert_attrs)
        self.intensity_expert = "intensity" in expert_attrs
        self.scale_slots = G if "scale" in expert_attrs else 1
        self.rot_slots = G if "rot" in expert_attrs else 1
        self.num_variables = num_variables
        # The parent's intensity tensor has one column per stored intensity.
        super().__init__(x_resol, y_resol, z_resol, num_channels=G if self.intensity_expert else num_variables)
        self.num_shapes = G
        self._route = torch.empty(0)

    # ---- shapes -------------------------------------------------------------
    @property
    def get_shape_scaling(self):
        return self.scaling_activation(self._scaling)  # [N, scale_slots, 3]

    @property
    def get_shape_rotation(self):
        return torch.nn.functional.normalize(self._rotation, dim=-1)  # [N, rot_slots, 4]

    def _envelope_index(self):
        return torch.prod(self.get_shape_scaling, dim=-1).argmax(dim=1)

    @property
    def get_scaling(self):
        index = self._envelope_index()
        return self.get_shape_scaling[torch.arange(index.shape[0], device=index.device), index]

    @property
    def get_rotation(self):
        index = self._envelope_index() if self.rot_slots == self.scale_slots else \
            torch.zeros(self._xyz.shape[0], dtype=torch.long, device="cuda")
        return self.get_shape_rotation[torch.arange(index.shape[0], device=index.device), index]

    def get_covariance(self, scaling_modifier=1):
        raise NotImplementedError("AttrMoEGaussianModel needs compute_cov3D_python=False")

    # ---- routing ------------------------------------------------------------
    def hard_route(self):
        return self._route.argmax(dim=-1)  # [N, V]

    def slot_intensity(self, v):
        """Intensity each slot would give variable v: [N, G]."""
        if self.intensity_expert:
            return self.get_opacity
        return self.get_opacity[:, v:v + 1].expand(-1, self.num_shapes)

    def variable_intensity(self, choice):
        """Intensity each variable gets from its chosen slot (choice [N, V]): [N, V]."""
        if self.intensity_expert:
            return torch.gather(self.get_opacity, 1, choice)
        return self.get_opacity

    def render_inputs(self, v, tau, hard):
        """Means, scales, rotations and densities to voxelize for variable v."""
        N, G = self._xyz.shape[0], self.num_shapes
        rows = torch.arange(N, device="cuda")
        if hard:
            slot = self.hard_route()[:, v]
            scale_slot = slot if self.scale_slots > 1 else torch.zeros_like(slot)
            rot_slot = slot if self.rot_slots > 1 else torch.zeros_like(slot)
            density = self.slot_intensity(v)[rows, slot].unsqueeze(-1)
            return (self.get_xyz, self.get_shape_scaling[rows, scale_slot],
                    self.get_shape_rotation[rows, rot_slot], density, 1)
        weights = torch.softmax(self._route[:, v, :] / tau, dim=-1)  # [N, G]
        routed = weights * self.slot_intensity(v)
        if self.scale_slots == 1 and self.rot_slots == 1:
            # Intensity-only experts: every slot has the same shape, so they collapse to one Gaussian.
            return self.get_xyz, self.get_shape_scaling[:, 0], self.get_shape_rotation[:, 0], routed.sum(-1, keepdim=True), 1
        means = self.get_xyz.unsqueeze(1).expand(N, G, 3).reshape(N * G, 3)
        scales = self.get_shape_scaling.expand(N, G, 3).reshape(N * G, 3)
        rotations = self.get_shape_rotation.expand(N, G, 4).reshape(N * G, 4)
        return means, scales, rotations, routed.reshape(N * G, 1), G

    def shape_usage(self, tau):
        """Group-lasso term per (primitive, slot): L2 norm over variables of routed intensity."""
        weights = torch.softmax(self._route / tau, dim=-1)  # [N, V, G]
        if self.intensity_expert:
            intensity = self.get_opacity.unsqueeze(1)  # [N, 1, G]
        else:
            intensity = self.get_opacity.unsqueeze(-1)  # [N, V, 1]
        routed = weights * intensity
        return torch.sqrt((routed ** 2).sum(dim=1) + 1e-12)  # [N, G]

    # ---- setup --------------------------------------------------------------
    def create_random_onthefly(self, num_start, spatial_lr_scale, chunk_len, seed=42, knn_on_scaled=False,
                               home=None, home_prob=0.9):
        super().create_random_onthefly(num_start, spatial_lr_scale, chunk_len, seed=seed, knn_on_scaled=knn_on_scaled)
        G = self.num_shapes
        N = self._xyz.shape[0]
        self._scaling = nn.Parameter(self._scaling.detach().unsqueeze(1).repeat(1, self.scale_slots, 1).requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation.detach().unsqueeze(1).repeat(1, self.rot_slots, 1).requires_grad_(True))
        # Each variable starts with home_prob on its home slot (hard routing as init).
        route = torch.zeros((N, self.num_variables, G), device="cuda")
        if G > 1:
            home_logit = math.log(home_prob * (G - 1) / (1.0 - home_prob))
            for v, g in enumerate(home):
                route[:, v, g] = home_logit
        self._route = nn.Parameter(route.requires_grad_(True))

    def training_setup(self, training_args, route_lr=0.01):
        super().training_setup(training_args)
        self.optimizer.add_param_group({"params": [self._route], "lr": route_lr, "name": "route"})

    # ---- densification ------------------------------------------------------
    def prune_points(self, mask):
        super().prune_points(mask)
        self._route = self._optimizable_by_name("route")

    def _optimizable_by_name(self, name):
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                return group["params"][0]
        raise KeyError(name)

    def densification_postfix_attr(self, new_xyz, new_opacities, new_scaling, new_rotation, new_route):
        optimizable = self.cat_tensors_to_optimizer({
            "xyz": new_xyz, "opacity": new_opacities, "scaling": new_scaling,
            "rotation": new_rotation, "route": new_route,
        })
        self._xyz = optimizable["xyz"]
        self._opacity = optimizable["opacity"]
        self._scaling = optimizable["scaling"]
        self._rotation = optimizable["rotation"]
        self._route = optimizable["route"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _sample_along_major_axis(self, selected, repeat):
        """Offsets along the largest axis of each selected primitive's envelope shape."""
        stds = self.get_scaling[selected].repeat(repeat, 1)
        major = stds.argmax(dim=1)
        stds_major = torch.zeros_like(stds)
        rows = torch.arange(stds.size(0), device=stds.device)
        stds_major[rows, major] = stds[rows, major]
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds_major)
        rots = build_rotation(self.get_rotation[selected]).repeat(repeat, 1, 1)
        return torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected].repeat(repeat, 1)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected = (padded_grad >= grad_threshold) & \
            (torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        new_xyz = self._sample_along_major_axis(selected, N)
        # Every scale slot shrinks by the same factor, as in single-shape ESVR.
        new_scaling = self.scaling_inverse_activation(self.get_shape_scaling[selected].repeat(N, 1, 1) / (0.8 * N))
        new_rotation = self._rotation[selected].repeat(N, 1, 1)
        new_opacity = self._opacity[selected].repeat(N, 1)
        new_route = self._route[selected].repeat(N, 1, 1)
        self.densification_postfix_attr(new_xyz, new_opacity, new_scaling, new_rotation, new_route)

        prune_filter = torch.cat((selected, torch.zeros(N * selected.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected = (torch.norm(grads, dim=-1) >= grad_threshold) & \
            (torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        new_xyz = self._sample_along_major_axis(selected, 1)
        self.densification_postfix_attr(new_xyz, self._opacity[selected], self._scaling[selected],
                                        self._rotation[selected], self._route[selected])

    # ---- storage ------------------------------------------------------------
    @torch.no_grad()
    def storage(self, hard):
        """Stored floats/bytes. Hard: only active intensities and the slots they use."""
        N, V, G = self._xyz.shape[0], self.num_variables, self.num_shapes
        per_primitive = CENTER_FLOATS + SCALE_FLOATS * self.scale_slots + ROT_FLOATS * self.rot_slots + self.num_channels
        dense_floats = N * per_primitive
        if not hard:
            return {"floats": dense_floats, "bytes": dense_floats * 4, "dense_bytes": dense_floats * 4,
                    "active_intensity_frac": 1.0, "shapes_per_primitive": float(G)}
        choice = self.hard_route()
        active = self.variable_intensity(choice) >= INTENSITY_MIN  # [N, V]
        rows = torch.arange(N, device="cuda").unsqueeze(1).expand(N, V)
        used = torch.zeros((N, G), dtype=torch.bool, device="cuda")
        used[rows[active], choice[active]] = True
        used_slots = int(used.sum())
        floats = N * CENTER_FLOATS
        floats += SCALE_FLOATS * (used_slots if self.scale_slots > 1 else N)
        floats += ROT_FLOATS * (used_slots if self.rot_slots > 1 else N)
        floats += used_slots if self.intensity_expert else int(active.sum())
        # Per primitive: V presence bits and V slot indices.
        index_bits = N * V * (1 + math.ceil(math.log2(G))) if G > 1 else N * V
        # The dense layout needs no index bits, so storage picks whichever is smaller.
        sparse_bytes = floats * 4 + index_bits / 8
        return {"floats": floats, "bytes": min(sparse_bytes, dense_floats * 4), "dense_bytes": dense_floats * 4,
                "active_intensity_frac": float(active.float().mean()),
                "shapes_per_primitive": float(used.sum(dim=1).float().mean())}

    def save_npz(self, path, hard):
        import numpy as np
        np.savez_compressed(
            path,
            xyz=self._xyz.detach().cpu().numpy(),
            scaling=self._scaling.detach().cpu().numpy(),
            rotation=self._rotation.detach().cpu().numpy(),
            intensity_logit=self._opacity.detach().cpu().numpy(),
            route_logit=self._route.detach().cpu().numpy(),
            expert_attrs=np.array(self.expert_attrs),
            hard=hard,
        )
