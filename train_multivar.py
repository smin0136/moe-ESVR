"""Multivariate ESVR with geometry experts.

Each expert is one GaussianModel (one geometry bank). The variables routed to an
expert share its xyz/scale/rotation and keep their own intensity channel.

  --groups shared     one expert for all variables (fully shared geometry)
  --groups separate   one expert per variable (per-variable ESVR)
  --groups g.json     {"groups": [["p", "rhol"], ...]} from cluster_variables.py

Routing modes:
  hard     each variable uses exactly the expert of its group.
  channel  every expert holds a channel for every variable; a variable is the
           sum of all experts whose (expert, variable) channel is active. A
           cost-weighted group-sparsity penalty drives unhelpful channels to
           zero, and they are switched off at the end of the penalty window.
           The routing gate of (k, v) is the RMS intensity of that channel,
           i.e. a weight-normalized gate.

Constraints for channel routing (both off by default):
  --topk k          a variable keeps at most k experts. mode "exact" keeps its
                    top-k channels by gate regardless of the threshold
                    (always k when they still render), mode "max" applies the
                    threshold first and then keeps at most k (1..k).
  --expert_cap N    each expert is pruned to at most N primitives (or
  --target_mb T     T MB in total, split equally over experts, counting every
                    channel as stored) at --budget_scan, by the global-pruning
                    importance score; this replaces the fixed-ratio global
                    prune at that point.
"""
import json
import math
import os
import random
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from dataset_configs import DATASET_CONFIGS
from gaussian_renderer import query_experts
from prune import compute_fa, prune_list, prune_zeros
from scene import GaussianModel
from utils.general_utils import inverse_sigmoid, safe_state
from utils.loss_utils import l1_loss, ssim
from utils.train_utils import (configure_scan_schedule, load_multivar_volume, prepare_output_and_logger,
                               write_volume_json)

# Same optimizer scale that Scene uses for single-variable ESVR.
SPATIAL_LR_SCALE = 11.02160816192627
SPATIAL_EXTENT = 2.0
# Per primitive: xyz(3) + scale(3) + rotation(4) + one float per channel.
GEOMETRY_FLOATS = 10
# Raw intensity written into switched-off channels (sigmoid ~ 2e-9).
OFF_LOGIT = -20.0
# The voxelizer skips contributions with alpha < 1/255 in forward and backward,
# so a channel below this everywhere renders nothing and receives no gradient.
VOXELIZER_ALPHA_MIN = 1.0 / 255.0


def resolve_groups(spec, variables):
    if spec == "shared":
        groups = [list(variables)]
    elif spec == "separate":
        groups = [[name] for name in variables]
    else:
        with open(spec) as f:
            loaded = json.load(f)
        groups = loaded["groups"] if isinstance(loaded, dict) else loaded

    flat = [name for group in groups for name in group]
    if sorted(flat) != sorted(variables):
        raise ValueError(f"Groups {groups} must cover each of {variables} exactly once")
    return groups


def axis_centers(size, patch_size):
    half = patch_size // 2
    count = math.ceil(size / patch_size)
    return [min(size - half, index * patch_size + half) for index in range(count)]


def psnr_from_mse(mse):
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


class Router:
    """Which (expert, channel) pairs reconstruct each variable."""

    def __init__(self, mode, groups, variables):
        self.mode = mode
        self.groups = groups
        self.variables = variables
        K, V = len(groups), len(variables)
        if mode == "hard":
            self.channels_per_expert = [len(g) for g in groups]
            # active[k][c] for the local channels of expert k
            self.active = [[True] * len(g) for g in groups]
            self.pairs = {name: [(k, c)] for k, g in enumerate(groups) for c, name in enumerate(g)}
        else:
            self.channels_per_expert = [V] * K
            self.active = [[True] * V for _ in range(K)]
            self.home = {name: k for k, g in enumerate(groups) for name in g}

    def members(self, v):
        if self.mode == "hard":
            return [(k, c) for k, c in self.pairs[self.variables[v]] if self.active[k][c]]
        return [(k, v) for k in range(len(self.groups)) if self.active[k][v]]

    def active_channels(self, k):
        return sum(self.active[k])

    def table(self):
        if self.mode == "hard":
            return {name: [list(kc) for kc in self.pairs[name]] for name in self.variables}
        return {name: [[k, v] for k, v in self.members(v)] for v, name in enumerate(self.variables)}


def channel_rms(expert):
    return torch.sqrt((expert.get_opacity ** 2).mean(dim=0) + 1e-12)


def model_size_mb(experts, router):
    floats = sum(
        e.get_xyz.shape[0] * (GEOMETRY_FLOATS + router.active_channels(k))
        for k, e in enumerate(experts) if router.active_channels(k) > 0
    )
    return floats * 4 / 1024**2


@torch.no_grad()
def evaluate(experts, router, variables, gt_vol, mask, pipe):
    """Full-volume PSNR per variable, plus the overall PSNR of the pooled MSE."""
    X, Y, Z = gt_vol.shape[1:]
    full_sVoxel = [SPATIAL_EXTENT, Y / X * SPATIAL_EXTENT, Z / X * SPATIAL_EXTENT]
    result = {"psnr": {}, "psnr_mask": {}}
    mse_all, mse_mask_all = [], []
    for v, name in enumerate(variables):
        members = router.members(v)
        pred = query_experts([experts[k] for k, _ in members], [c for _, c in members],
                             [0, 0, 0], [X, Y, Z], full_sVoxel, pipe)["vol"]
        pred = torch.clamp(pred.reshape(X, Y, Z), 0.0, 1.0)
        sq_err = (pred - gt_vol[v]) ** 2
        mse = sq_err.mean().item()
        mse_all.append(mse)
        result["psnr"][name] = psnr_from_mse(mse)
        if mask is not None:
            mse_mask = sq_err[mask].mean().item()
            mse_mask_all.append(mse_mask)
            result["psnr_mask"][name] = psnr_from_mse(mse_mask)

    # Every variable has the same voxel count, so pooled MSE is the mean MSE.
    result["overall_psnr"] = psnr_from_mse(sum(mse_all) / len(mse_all))
    result["mean_psnr"] = sum(result["psnr"].values()) / len(variables)
    if mask is not None:
        result["overall_psnr_mask"] = psnr_from_mse(sum(mse_mask_all) / len(mse_mask_all))
    result["num_primitives"] = [e.get_xyz.shape[0] if router.active_channels(k) > 0 else 0
                                for k, e in enumerate(experts)]
    result["size_mb"] = model_size_mb(experts, router)
    result["routing"] = router.table()
    if router.mode == "channel":
        result["gate_rms"] = [channel_rms(e).tolist() for e in experts]
    return result


def format_eval(result, variables, router):
    lines = ["  PSNR      : " + "  ".join(f"{n}={result['psnr'][n]:.2f}" for n in variables)]
    if result["psnr_mask"]:
        lines.append("  PSNR(mask): " + "  ".join(f"{n}={result['psnr_mask'][n]:.2f}" for n in variables))
    overall = f"  overall={result['overall_psnr']:.2f} dB  mean={result['mean_psnr']:.2f} dB"
    if "overall_psnr_mask" in result:
        overall += f"  overall(mask)={result['overall_psnr_mask']:.2f} dB"
    lines.append(overall)
    lines.append(f"  primitives={result['num_primitives']}  total={sum(result['num_primitives'])}  size={result['size_mb']:.3f} MB")
    if router.mode == "channel":
        lines.append("  gate RMS (row=expert, * = active): " + " ".join(f"{n:>9s}" for n in variables))
        for k, row in enumerate(result["gate_rms"]):
            cells = [f"{g:8.5f}{'*' if router.active[k][v] else ' '}" for v, g in enumerate(row)]
            lines.append(f"    E{k}: " + " ".join(cells))
    return "\n".join(lines)


def training(dataset, opt, pipe, args):
    config = DATASET_CONFIGS[args.dataset_name]
    variables = config["variables"]
    if args.uniform_init > 0:
        # No prior assignment: K identical-role experts with every channel on.
        if args.routing != "channel":
            raise ValueError("--uniform_init needs --routing channel")
        groups = [list(variables) for _ in range(args.uniform_init)]
    elif args.fixed_routing:
        # Retrain on a routing learned by an earlier run: its groups give the
        # experts, its final (expert, variable) pairs stay on and the rest off.
        source = os.path.dirname(args.fixed_routing)
        groups = json.load(open(os.path.join(source, "routing.json")))["groups"]
        fixed_table = json.load(open(args.fixed_routing))["routing"]
        if args.routing != "channel":
            raise ValueError("--fixed_routing needs --routing channel")
    else:
        groups = resolve_groups(args.groups, variables)
    router = Router(args.routing, groups, variables)
    if args.fixed_routing:
        router.active = [[[k, v] in fixed_table[name] for v, name in enumerate(variables)] for k in range(len(groups))]
        print(f"Fixed routing from {args.fixed_routing}: {router.table()}")
    K, V = len(groups), len(variables)

    prepare_output_and_logger(dataset)
    model_path = dataset.model_path

    gt_vol, mask, normalization = load_multivar_volume(config, args.timestep)
    _, X, Y, Z = gt_vol.shape
    write_volume_json(model_path, (X, Y, Z))
    print(f"Variables: {variables}")
    print(f"Expert groups ({K}): {groups}  routing={args.routing}  densify_stat={args.densify_stat}")

    iters_per_scan = configure_scan_schedule(
        opt, args, args.subvol_size, (X, Y, Z), os.path.join(model_path, "scan_conversion_log.txt")
    )
    eval_iters = {s * iters_per_scan for s in args.eval_scans}
    lasso_start, lasso_end = (s * iters_per_scan for s in args.lasso_scans)
    topk_iter = (args.topk_scan * iters_per_scan) if args.topk_scan is not None else lasso_end
    pool_budget = args.budget_mode in ("exact", "pool", "usage", "ratio") and args.target_mb > 0
    if args.budget_scan is not None:
        budget_iter = args.budget_scan * iters_per_scan
    elif pool_budget:
        # After the routing is fixed, so the budget counts only the channels that stay.
        budget_iter = lasso_end
    else:
        budget_iter = opt.globalpruning_iter[0] if opt.globalpruning_iter else None

    run_info = {"dataset_name": args.dataset_name, "variables": variables, "groups": groups,
                "routing_mode": args.routing, "routing": router.table(),
                "timestep": args.timestep if args.timestep is not None else config["timestep"],
                "normalization": normalization, "args": vars(args)}
    with open(os.path.join(model_path, "routing.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    # Patches are clipped per axis so thin volumes (e.g. 512x64x32) still fit.
    patch = [min(args.subvol_size, d) for d in (X, Y, Z)]
    half = [p // 2 for p in patch]
    subvol_centers = [
        (cx, cy, cz)
        for cx in axis_centers(X, patch[0])
        for cy in axis_centers(Y, patch[1])
        for cz in axis_centers(Z, patch[2])
    ]
    x_normalized = torch.linspace(-SPATIAL_EXTENT / 2, SPATIAL_EXTENT / 2 - 2 / X, steps=X)
    y_normalized = x_normalized[(X - Y) // 2:(X - Y) // 2 + Y]
    z_normalized = x_normalized[(X - Z) // 2:(X - Z) // 2 + Z]
    tv_vol_nVoxel = torch.tensor(patch)
    tv_vol_sVoxel = torch.tensor(SPATIAL_EXTENT / X) * tv_vol_nVoxel
    print(f"Patch {patch}, subvolumes per scan: {len(subvol_centers)}")

    # With --split_budget, init_num and max_primitives are totals shared by all
    # experts in proportion to their (initial) group sizes.
    share = [(1.0 / K if args.uniform_init > 0 else len(g) / V) if args.split_budget else 1.0 for g in groups]
    init_counts = [max(1, round(opt.init_num * s)) for s in share]
    caps = [round(args.max_primitives * s) for s in share]
    print(f"Init per expert: {init_counts}  cap per expert: {caps}")

    # Final primitive budget per expert (0 = none).
    if pool_budget:
        budget_caps = [0] * K
        print(f"Budget ({args.budget_mode}): {args.target_mb} MB over all experts at iteration {budget_iter}"
              " (replaces global pruning)")
    elif args.expert_target:
        # Each expert grows to target / remainweight, then is pruned to exactly its
        # target at the budget scan (as the global prune would cut a grown model).
        budget_caps = [t if router.active_channels(k) > 0 else 0 for k, t in enumerate(args.expert_target)]
        caps = [math.ceil(t / args.pruning_remainweight) for t in budget_caps]
        print(f"Expert size targets {budget_caps}, grown to {caps} before the prune")
    elif args.target_mb > 0:
        budget_caps = [round(args.target_mb * 1024**2 / 4 / K / (GEOMETRY_FLOATS + router.channels_per_expert[k]))
                       for k in range(K)]
    else:
        budget_caps = [args.expert_cap] * K
    if any(c > 0 for c in budget_caps):
        if budget_iter is None:
            raise ValueError("A primitive budget needs --budget_scan when there is no global pruning")
        print(f"Primitive budget per expert: {budget_caps} at iteration {budget_iter}")

    experts = []
    for k, group in enumerate(groups):
        expert = GaussianModel(X, Y, Z, num_channels=router.channels_per_expert[k])
        # Uniform experts get their own init points so they are not exact copies.
        seed = 42 + args.seed + (1000 * k if args.uniform_init > 0 else 0)
        expert.create_random_onthefly(init_counts[k], args.spatial_lr_scale, SPATIAL_EXTENT, seed=seed,
                                      knn_on_scaled=args.init_knn_scaled)
        if args.routing == "channel" and args.uniform_init == 0 and not args.fixed_routing:
            # Channels outside the initial group start nearly off (hard routing as init).
            with torch.no_grad():
                for v, name in enumerate(variables):
                    if router.home[name] != k:
                        expert._opacity[:, v] = inverse_sigmoid(torch.tensor(args.off_init)).item()
        expert.training_setup(opt)
        experts.append(expert)

    # Per-expert scale bounds (min:max of every axis in voxels, 0 = none), kept by
    # clamping the log-scale after each update.
    scale_bounds = [(None, None)] * K
    if args.scale_bounds:
        if len(args.scale_bounds) != K:
            raise ValueError(f"--scale_bounds needs one min:max per expert ({K})")
        voxel = SPATIAL_EXTENT / X
        scale_bounds = [tuple(math.log(float(x) * voxel) if float(x) > 0 else None for x in b.split(":"))
                        for b in args.scale_bounds]
        print(f"Scale bounds per expert (voxels): {args.scale_bounds}")

    def clamp_scales(k):
        lo, hi = scale_bounds[k]
        if lo is not None or hi is not None:
            with torch.no_grad():
                experts[k]._scaling.clamp_(min=lo, max=hi)

    for k in range(K):
        clamp_scales(k)

    # Progressive training: expert k joins at its start scan. Until then it adds
    # nothing to the volume and is not updated; it joins with low intensity, so
    # it learns the residual of the experts already trained.
    start_iters = [s * iters_per_scan for s in args.progressive_scans] if args.progressive_scans else [0] * K
    if len(start_iters) != K:
        raise ValueError(f"--progressive_scans needs one start scan per expert ({K})")
    with torch.no_grad():
        for k, e in enumerate(experts):
            if start_iters[k] > 0:
                e._opacity[:] = OFF_LOGIT
    if args.progressive_scans:
        print(f"Progressive start scans per expert: {args.progressive_scans}")

    # Loss weight per variable. Hard: each expert gets the mean of its variables,
    # so a one-variable expert matches single-variable ESVR. Channel: plain mean.
    if args.routing == "hard":
        weights = [1.0 / len(groups[router.pairs[name][0][0]]) for name in variables]
    else:
        weights = [1.0 / V] * V

    log_path = os.path.join(model_path, "psnr_log.tsv")
    with open(log_path, "w") as f:
        header = ["iteration", "scan", "loss"] + [f"psnr_{n}" for n in variables]
        header += ["overall_psnr", "mean_psnr"]
        if mask is not None:
            header += [f"psnr_mask_{n}" for n in variables] + ["overall_psnr_mask"]
        header += ["num_primitives", "size_mb", "elapsed_s"]
        f.write("\t".join(header) + "\n")

    def log_eval(iteration, loss_value, elapsed):
        result = evaluate(experts, router, variables, gt_vol, mask, pipe)
        print(f"\n[ITER {iteration} | scan {iteration // iters_per_scan}] loss={loss_value:.6f} elapsed={elapsed / 60:.1f} min")
        print(format_eval(result, variables, router))
        row = [iteration, iteration // iters_per_scan, f"{loss_value:.6f}"]
        row += [f"{result['psnr'][n]:.4f}" for n in variables]
        row += [f"{result['overall_psnr']:.4f}", f"{result['mean_psnr']:.4f}"]
        if mask is not None:
            row += [f"{result['psnr_mask'][n]:.4f}" for n in variables] + [f"{result['overall_psnr_mask']:.4f}"]
        row += [",".join(map(str, result["num_primitives"])), f"{result['size_mb']:.4f}", f"{elapsed:.1f}"]
        with open(log_path, "a") as f:
            f.write("\t".join(map(str, row)) + "\n")
        return result

    def switch_off_channels(iteration, threshold=True):
        """Turn off channels whose gate fell below threshold and/or outside each variable's top-k."""
        with torch.no_grad():
            # An expert without primitives has no gate and no peak.
            rms = torch.stack([channel_rms(e) if e.get_xyz.shape[0] > 0 else torch.zeros(V, device="cuda")
                               for e in experts])  # [K, V]
            peak = torch.stack([e.get_opacity.max(dim=0).values if e.get_xyz.shape[0] > 0
                                else torch.zeros(V, device="cuda") for e in experts])
            active = torch.tensor(router.active, device="cuda")
            rms = torch.where(active, rms, torch.zeros_like(rms))
            renders = active & (peak >= VOXELIZER_ALPHA_MIN)
            keep = renders & (rms >= args.channel_threshold) if threshold else renders
            if args.topk > 0:
                rank = rms.argsort(dim=0, descending=True).argsort(dim=0)  # 0 = strongest expert
                in_topk = rank < args.topk
                keep = (renders & in_topk) if args.topk_mode == "exact" else (keep & in_topk)
            # Every variable keeps at least its strongest expert.
            keep[rms.argmax(dim=0), torch.arange(V)] = True
            for k in range(K):
                for v in range(V):
                    router.active[k][v] = bool(keep[k, v])
        print(f"\n[ITER {iteration}] channel routing fixed: {router.table()}")

    def prune_to_pooled_budget(iteration):
        """Prune all experts together to exactly --target_mb, counting a primitive
        of expert k as its geometry plus the expert's active channels.
        exact: every expert gets an equal share of the floats; what an expert
               cannot use goes to the others (water-filling). Each keeps its
               most important primitives.
        usage: as exact, but shares are proportional to the expert's usage,
               the sum of its active gates (channel RMS).
        ratio: as exact, but shares follow --budget_ratio, the largest ratio
               going to the most used expert.
        pool:  the primitives with the most importance per stored float are
               kept, over all experts together."""
        with torch.no_grad():
            usage = {k: float(channel_rms(e)[torch.tensor(router.active[k], device="cuda")].sum())
                     for k, e in enumerate(experts) if router.active_channels(k) > 0 and e.get_xyz.shape[0] > 0}
            live = []
            for k, e in enumerate(experts):
                if router.active_channels(k) == 0 or e.get_xyz.shape[0] < 2:
                    continue
                prune_zeros(e, None, pipe, None, X, Y, Z, SPATIAL_EXTENT)
                e.prune_dark_gaussians()
                if e.get_xyz.shape[0] >= 2:
                    live.append(k)
            scores, costs, owners = [], [], []
            for k in live:
                e = experts[k]
                _, imp_list, _ = prune_list(e, None, pipe, None, X, Y, Z, SPATIAL_EXTENT)
                cost = GEOMETRY_FLOATS + router.active_channels(k)
                score = imp_list.reshape(-1) * compute_fa(e.get_scaling).reshape(-1)
                scores.append(score / cost)
                costs.append(torch.full_like(score, cost))
                owners.append(torch.full(score.shape, k, dtype=torch.long, device=score.device))
            score, cost, owner = torch.cat(scores), torch.cat(costs), torch.cat(owners)
            budget_floats = args.target_mb * 1024**2 / 4
            keep = torch.zeros_like(score, dtype=torch.bool)
            if args.budget_mode == "pool":
                order = score.argsort(descending=True)
                keep[order] = cost[order].cumsum(0) <= budget_floats
            else:
                if args.budget_mode == "usage":
                    weight = {k: usage[k] for k in live}
                elif args.budget_mode == "ratio":
                    ratios = sorted(args.budget_ratio, reverse=True)
                    ratios += [ratios[-1]] * max(0, len(live) - len(ratios))
                    weight = {k: ratios[i] for i, k in enumerate(sorted(live, key=lambda k: -usage[k]))}
                else:
                    weight = {k: 1.0 for k in live}
                floats = {k: experts[k].get_xyz.shape[0] * (GEOMETRY_FLOATS + router.active_channels(k)) for k in live}
                share, filling = {}, list(live)
                while filling:
                    total = sum(weight[k] for k in filling)
                    share = {k: budget_floats * weight[k] / total for k in filling}
                    small = [k for k in filling if floats[k] <= share[k]]
                    if not small:
                        break
                    budget_floats -= sum(floats[k] for k in small)
                    filling = [k for k in filling if k not in small]
                print(f"\n[ITER {iteration}] budget weights {({k: round(weight[k], 4) for k in live})}")
                for k in live:
                    idx = torch.nonzero(owner == k).squeeze(1)
                    n_keep = idx.numel() if k not in filling else int(share[k] // (GEOMETRY_FLOATS + router.active_channels(k)))
                    keep[idx[score[idx].argsort(descending=True)[:n_keep]]] = True
            for k in live:
                kept = keep[owner == k]
                if int(kept.sum()) < 2:
                    # Pruning helpers need two primitives; an expert the budget
                    # leaves (almost) empty is retired instead.
                    router.active[k] = [False] * len(router.active[k])
                    caps[k] = 1
                    continue
                experts[k].prune_points(~kept)
                caps[k] = experts[k].get_xyz.shape[0]
        print(f"\n[ITER {iteration}] budget ({args.budget_mode}) {args.target_mb} MB: primitives per expert "
              f"{[e.get_xyz.shape[0] if router.active_channels(k) > 0 else 0 for k, e in enumerate(experts)]}, "
              f"size {model_size_mb(experts, router):.3f} MB")

    def silence_off_channels():
        with torch.no_grad():
            for k, expert in enumerate(experts):
                off = [c for c, on in enumerate(router.active[k]) if not on]
                if off:
                    expert._opacity[:, off] = OFF_LOGIT

    start_time = time.time()
    subvol_ptr = 0
    ema_loss = 0.0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):
        scan_aligned_iter = (iteration // iters_per_scan) * iters_per_scan
        for k, expert in enumerate(experts):
            # A late expert runs its own learning-rate schedule from its start.
            expert.update_learning_rate(max(0, scan_aligned_iter - start_iters[k]))
            if iteration == 1:
                expert.prune_out_bbox(SPATIAL_EXTENT)
            if iteration == start_iters[k] + 1 and start_iters[k] > 0:
                with torch.no_grad():
                    expert._opacity[:] = inverse_sigmoid(torch.tensor(args.off_init)).item()
                print(f"\n[ITER {iteration}] expert {k} joins training")

        cx, cy, cz = subvol_centers[subvol_ptr]
        subvol_ptr = (subvol_ptr + 1) % len(subvol_centers)
        tv_vol_center = (x_normalized[cx], y_normalized[cy], z_normalized[cz])
        gt_subvol = gt_vol[
            :,
            cx - half[0]:cx + half[0],
            cy - half[1]:cy + half[1],
            cz - half[2]:cz + half[2],
        ]

        # One backward per variable. The xyz gradient a variable adds to each
        # expert gives its own densification signal; densify_stat=max keeps the
        # largest one, so variables cannot cancel each other's gradients.
        grad_stat = [torch.zeros((e.get_xyz.shape[0], 1), device="cuda") for e in experts]
        visibility = [torch.zeros(e.get_xyz.shape[0], dtype=torch.bool, device="cuda") for e in experts]
        radii_max = [torch.zeros(e.get_xyz.shape[0], device="cuda") for e in experts]
        total_loss = 0.0
        for v in range(V):
            members = [(k, c) for k, c in router.members(v) if iteration > start_iters[k]]
            query_pkg = query_experts([experts[k] for k, _ in members], [c for _, c in members],
                                      tv_vol_center, tv_vol_nVoxel, tv_vol_sVoxel, pipe)
            vol_pred = query_pkg["vol"].reshape(gt_subvol.shape[1:])
            gt = gt_subvol[v]
            loss_v = (1.0 - opt.lambda_dssim) * l1_loss(vol_pred, gt) + opt.lambda_dssim * (1.0 - ssim(vol_pred, gt))
            total_loss += weights[v] * loss_v.item()

            previous = [experts[k]._xyz.grad.clone() if experts[k]._xyz.grad is not None else None for k, _ in members]
            (weights[v] * loss_v).backward()
            for j, (k, _) in enumerate(members):
                grad = experts[k]._xyz.grad if previous[j] is None else experts[k]._xyz.grad - previous[j]
                norm = torch.norm(grad[:, :3], dim=-1, keepdim=True)
                if args.densify_stat == "max":
                    grad_stat[k] = torch.maximum(grad_stat[k], norm / weights[v])
                visibility[k] |= query_pkg["visibility_filter"][j]
                radii_max[k] = torch.maximum(radii_max[k], query_pkg["radii"][j].float())

        if args.densify_stat == "mean":
            # Norm of the summed gradient (the original single-backward behaviour).
            for k, e in enumerate(experts):
                if e._xyz.grad is not None:
                    grad_stat[k] = torch.norm(e._xyz.grad[:, :3], dim=-1, keepdim=True)

        if args.routing == "channel" and args.lasso_lambda > 0 and lasso_start <= iteration < lasso_end:
            # Cost-weighted group sparsity: a channel costs N_k floats to store.
            sizes = torch.tensor([float(e.get_xyz.shape[0]) for e in experts], device="cuda")
            cost = sizes / sizes.mean()
            penalty = sum(cost[k] * channel_rms(e)[torch.tensor(router.active[k], device="cuda")].sum()
                          for k, e in enumerate(experts) if iteration > start_iters[k])
            (args.lasso_lambda * penalty).backward()

        with torch.no_grad():
            for k, expert in enumerate(experts):
                if iteration < opt.densify_until_iter and router.active_channels(k) > 0:
                    vis = visibility[k]
                    expert.max_radii2D[vis] = torch.max(expert.max_radii2D[vis], radii_max[k][vis])
                    expert.add_densification_stats_norm(grad_stat[k], vis)

            if iteration % 100 == 0:
                torch.cuda.empty_cache()

            for k, expert in enumerate(experts):
                if router.active_channels(k) == 0 or iteration <= start_iters[k]:
                    continue
                if expert.get_xyz.shape[0] == 0:
                    # Strong penalties can prune an expert away entirely; retire its channels.
                    router.active[k] = [False] * len(router.active[k])
                    print(f"\n[ITER {iteration}] expert {k} has no primitives left; its channels are off")
                    continue
                # Pruning helpers do not handle a single remaining primitive.
                prunable = expert.get_xyz.shape[0] >= 2
                if prunable and not args.no_pruning and (iteration in opt.darkpruning_iter or iteration == opt.iterations):
                    prune_zeros(expert, None, pipe, None, X, Y, Z, SPATIAL_EXTENT)
                    expert.prune_dark_gaussians()

                budget_now = budget_caps[k] > 0 and iteration == budget_iter
                global_prune = iteration in opt.globalpruning_iter and not args.no_pruning and not pool_budget
                if prunable and (global_prune or budget_now):
                    prune_zeros(expert, None, pipe, None, X, Y, Z, SPATIAL_EXTENT)
                    expert.prune_dark_gaussians()
                    _, imp_list, _ = prune_list(expert, None, pipe, None, X, Y, Z, SPATIAL_EXTENT)
                    score = imp_list * compute_fa(expert.get_scaling)
                    count = expert.get_xyz.shape[0]
                    if budget_now:
                        # Prune to the budget instead of by a fixed ratio.
                        if count > budget_caps[k]:
                            expert.prune_gaussians((count - budget_caps[k]) / count, score)
                        # Later densification may not grow the expert past its budget.
                        caps[k] = budget_caps[k] if caps[k] <= 0 else min(caps[k], budget_caps[k])
                    else:
                        expert.prune_gaussians(opt.pruning_remainweight, score)

                # As in mtvESVR, densification and opacity reset stop once an
                # expert reaches its primitive cap.
                below_cap = caps[k] <= 0 or expert.get_xyz.shape[0] < caps[k]
                if opt.densify_from_iter <= iteration <= opt.densify_until_iter and below_cap:
                    if iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        threshold = opt.densify_grad_threshold
                        if args.expert_target and caps[k] > 0:
                            # Lower the gradient threshold when needed so the expert
                            # reaches its size by the end of densification.
                            steps_left = (opt.densify_until_iter - iteration) // opt.densification_interval + 1
                            need = math.ceil((caps[k] - expert.get_xyz.shape[0]) / steps_left)
                            grads = (expert.xyz_gradient_accum / expert.denom).squeeze(-1).nan_to_num(0.0)
                            if 0 < need < grads.numel():
                                threshold = min(threshold, float(grads.topk(need).values[-1]))
                        expert.densify_and_prune(
                            threshold, 1 / 255, SPATIAL_EXTENT,
                            size_threshold, None, iteration, SPATIAL_EXTENT,
                        )
                        count = expert.get_xyz.shape[0]
                        if caps[k] > 0 and count > caps[k]:
                            # Trim the overshoot of the last densify step so every
                            # method ends with exactly its primitive budget.
                            expert.prune_gaussians((count - caps[k]) / count, expert.get_aggregate_intensity)
                    if iteration % opt.opacity_reset_interval == 0:
                        expert.reset_opacity()
                elif (opt.densify_from_iter <= iteration <= opt.densify_until_iter and args.cap_bbox_prune
                      and iteration % opt.densification_interval == 0):
                    # At the cap densify_and_prune is skipped; keep its box pruning
                    # so centers outside the volume are removed as in the original.
                    expert.prune_out_bbox(SPATIAL_EXTENT, margin=0.0, verbose=False)

                if iteration < opt.iterations:
                    expert.optimizer.step()
                    expert.optimizer.zero_grad(set_to_none=True)
                clamp_scales(k)

            if args.routing == "channel":
                if args.topk > 0 and iteration == topk_iter and topk_iter != lasso_end:
                    switch_off_channels(iteration, threshold=False)
                if iteration == lasso_end and not args.fixed_routing:
                    switch_off_channels(iteration)
                silence_off_channels()
            if pool_budget and iteration == budget_iter:
                prune_to_pooled_budget(iteration)

            ema_loss = 0.4 * total_loss + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.7f}", "N": sum(e.get_xyz.shape[0] for e in experts)})
                progress_bar.update(10)

            if iteration in eval_iters or iteration == opt.iterations:
                result = log_eval(iteration, ema_loss, time.time() - start_time)

    progress_bar.close()
    total_time = time.time() - start_time
    for k, expert in enumerate(experts):
        if router.active_channels(k) > 0:
            expert.save_ply(os.path.join(model_path, f"expert_{k}", "point_cloud.ply"))
    run_info["routing"] = router.table()
    run_info["active"] = router.active
    with open(os.path.join(model_path, "routing.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    result["train_time_s"] = total_time
    result["groups"] = groups
    result["routing_mode"] = args.routing
    with open(os.path.join(model_path, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Final] {args.dataset_name} groups={args.groups} routing={args.routing} time={total_time / 60:.1f} min")
    print(format_eval(result, variables, router))


if __name__ == "__main__":
    parser = ArgumentParser(description="Multivariate ESVR with geometry experts")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    multivar_datasets = sorted(n for n, c in DATASET_CONFIGS.items() if c.get("format") in ("npy_points", "raw_vars"))
    parser.add_argument("--dataset_name", type=str, required=True, choices=multivar_datasets)
    parser.add_argument("--groups", type=str, default="shared", help="shared | separate | path to groups json")
    parser.add_argument("--routing", type=str, default="hard", choices=["hard", "channel"])
    parser.add_argument("--fixed_routing", type=str, default="",
                        help="metrics.json of an earlier channel-routing run: retrain with its final routing fixed "
                             "from the start (its groups give the experts; no switch-off)")
    parser.add_argument("--expert_target", nargs="+", type=int, default=None,
                        help="Exact primitives per expert (one value per expert): grown to target/remainweight "
                             "during densification, pruned to target at the budget scan")
    parser.add_argument("--uniform_init", type=int, default=0,
                        help="Channel routing without prior groups: K experts, every channel on at init, "
                             "different init points per expert (overrides --groups; 0 = off)")
    parser.add_argument("--scale_bounds", nargs="+", type=str, default=None,
                        help="Per-expert scale bounds 'min:max' in voxels for every axis, 0 = none "
                             "(e.g. 6:0 0:0 0:3 = coarse / mid / fine)")
    parser.add_argument("--progressive_scans", nargs="+", type=int, default=None,
                        help="Start scan per expert (e.g. 0 100 200): later experts learn the residual")
    parser.add_argument("--densify_stat", type=str, default="max", choices=["max", "mean"],
                        help="max: largest per-variable xyz gradient; mean: gradient of the mean loss")
    parser.add_argument("--lasso_lambda", type=float, default=0.0, help="Channel group-sparsity weight")
    parser.add_argument("--lasso_scans", nargs=2, type=int, default=[400, 2000],
                        help="Penalty window in scans; channels are switched off at its end")
    parser.add_argument("--channel_threshold", type=float, default=1e-3, help="Gate RMS below which a channel is off")
    parser.add_argument("--off_init", type=float, default=0.02,
                        help="Initial intensity of out-of-group channels; must exceed 1/255 to receive gradient")
    parser.add_argument("--topk", type=int, default=0, help="Max experts per variable in channel routing (0 = no limit)")
    parser.add_argument("--topk_mode", type=str, default="max", choices=["max", "exact"],
                        help="exact: keep the top-k channels; max: threshold first, then at most k")
    parser.add_argument("--topk_scan", type=int, default=None,
                        help="Scan at which top-k is first enforced (default: end of the penalty window)")
    parser.add_argument("--no_cap_bbox_prune", dest="cap_bbox_prune", action="store_false",
                        help="Skip box pruning while an expert is at its primitive cap (behavior before 2026-10-29)")
    parser.add_argument("--expert_cap", type=int, default=0, help="Final primitive budget per expert (0 = none)")
    parser.add_argument("--target_mb", type=float, default=0.0,
                        help="Final total size budget in MB, split equally over experts (overrides --expert_cap)")
    parser.add_argument("--budget_scan", type=int, default=None,
                        help="Scan of the budget prune (default: the first global pruning; pool: end of the penalty window)")
    parser.add_argument("--budget_mode", type=str, default="split", choices=["split", "exact", "pool", "usage", "ratio"],
                        help="split: --target_mb split equally over experts at the global prune, counting every "
                             "channel (an upper bound); exact: after routing is fixed, an equal share per expert "
                             "counting active channels, unused shares redistributed; usage / ratio: as exact with "
                             "shares by summed active gate / by --budget_ratio; pool: one budget over all "
                             "experts by importance per stored float")
    parser.add_argument("--budget_ratio", nargs="+", type=float, default=[60, 30, 10],
                        help="Shares for --budget_mode ratio, assigned largest first to the experts ranked by usage; "
                             "experts beyond the list get its smallest value")
    parser.add_argument("--timestep", type=int, default=None, help="Overrides the config timestep")
    parser.add_argument("--subvol_size", type=int, default=128)
    parser.add_argument("--init_num", type=int, default=100_000, help="Initial Gaussians per expert")
    parser.add_argument("--pruning_remainweight", type=float, default=0.66)
    parser.add_argument("--darkpruning_iter", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--globalpruning_iter", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--eval_scans", nargs="+", type=int, default=[400, 1200, 2000], help="Scans at which to evaluate")
    parser.add_argument("--init_knn_scaled", action="store_true",
                        help="Initial scales from point spacing after aspect scaling (thin volumes)")
    parser.add_argument("--max_primitives", type=int, default=0, help="Stop densify/opacity reset once an expert reaches this many primitives (0 = off)")
    parser.add_argument("--split_budget", action="store_true",
                        help="Treat init_num and max_primitives as totals split across experts by group size")
    parser.add_argument("--spatial_lr_scale", type=float, default=SPATIAL_LR_SCALE,
                        help="Multiplies all learning rates (ESVR uses 11.02; mtvESVR effectively used 1.0)")
    parser.add_argument("--no_pruning", action="store_true", help="Disable zero/dark/global pruning")
    parser.add_argument("--seed", type=int, default=0, help="Offsets the init point seed and seeds torch/numpy")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    safe_state(args.quiet)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)
    print("\nTraining complete.")
