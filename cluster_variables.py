"""Variable affinity from a per-variable ESVR run, then hierarchical grouping.

Input is a `train_multivar.py --groups separate` run, where expert k holds the
geometry of variable k. Two affinities are computed:

  transfer  Freeze the geometry of variable a and fit only intensities for each
            variable b. drop[a, b] = PSNR(b on own geometry) - PSNR(b on a's geometry),
            where both PSNRs come from the same intensity-only refit.
  chamfer   Symmetric Chamfer distance between primitive centers of a and b.

Each matrix is symmetrized and clustered (average linkage) into K groups; the
groups are written as json files that train_multivar.py --groups accepts.
"""
import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from arguments import PipelineParams
from dataset_configs import DATASET_CONFIGS
from gaussian_renderer import query
from scene import GaussianModel
from utils.general_utils import inverse_sigmoid, safe_state
from utils.loss_utils import l1_loss, ssim
from utils.train_utils import load_multivar_volume
from train_multivar import SPATIAL_EXTENT, axis_centers, psnr_from_mse


def load_geometry(run_dir, routing, name, X, Y, Z):
    k, _ = routing[name]
    model = GaussianModel(X, Y, Z)
    model.load_ply(os.path.join(run_dir, f"expert_{k}", "point_cloud.ply"))
    return model


@torch.no_grad()
def channel_psnr(model, gt_vol, mask, pipe):
    V, X, Y, Z = gt_vol.shape
    full_sVoxel = [SPATIAL_EXTENT, Y / X * SPATIAL_EXTENT, Z / X * SPATIAL_EXTENT]
    psnr = []
    for b in range(V):
        pred = query(model, [0, 0, 0], [X, Y, Z], full_sVoxel, pipe, channel_idx=b)["vol"]
        sq_err = (torch.clamp(pred.reshape(X, Y, Z), 0.0, 1.0) - gt_vol[b]) ** 2
        psnr.append(psnr_from_mse((sq_err[mask] if mask is not None else sq_err).mean().item()))
    return psnr


def transfer_row(model, gt_vol, mask, pipe, args):
    """Fit all V intensity channels on a frozen geometry; return PSNR per target."""
    V, X, Y, Z = gt_vol.shape
    N = model.get_xyz.shape[0]
    for attr in ("_xyz", "_scaling", "_rotation"):
        getattr(model, attr).requires_grad_(False)
    model.num_channels = V
    model._opacity = torch.nn.Parameter(
        inverse_sigmoid(0.1 * torch.ones((N, V), dtype=torch.float, device="cuda")).requires_grad_(True)
    )
    optimizer = torch.optim.Adam([model._opacity], lr=args.intensity_lr, eps=1e-15)

    patch = [min(args.subvol_size, d) for d in (X, Y, Z)]
    half = [p // 2 for p in patch]
    centers = [(cx, cy, cz)
               for cx in axis_centers(X, patch[0])
               for cy in axis_centers(Y, patch[1])
               for cz in axis_centers(Z, patch[2])]
    x_normalized = torch.linspace(-SPATIAL_EXTENT / 2, SPATIAL_EXTENT / 2 - 2 / X, steps=X)
    y_normalized = x_normalized[(X - Y) // 2:(X - Y) // 2 + Y]
    z_normalized = x_normalized[(X - Z) // 2:(X - Z) // 2 + Z]
    nVoxel = torch.tensor(patch)
    sVoxel = torch.tensor(SPATIAL_EXTENT / X) * nVoxel

    for iteration in range(args.scans * len(centers)):
        cx, cy, cz = centers[iteration % len(centers)]
        gt_subvol = gt_vol[:, cx - half[0]:cx + half[0], cy - half[1]:cy + half[1], cz - half[2]:cz + half[2]]
        center = (x_normalized[cx], y_normalized[cy], z_normalized[cz])
        loss = 0.0
        for b in range(V):
            pred = query(model, center, nVoxel, sVoxel, pipe, channel_idx=b)["vol"].reshape(gt_subvol.shape[1:])
            loss = loss + (1.0 - args.lambda_dssim) * l1_loss(pred, gt_subvol[b]) \
                + args.lambda_dssim * (1.0 - ssim(pred, gt_subvol[b]))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return channel_psnr(model, gt_vol, mask, pipe)


@torch.no_grad()
def chamfer(points_a, points_b, chunk=8192):
    def one_way(src, dst):
        total = 0.0
        for start in range(0, src.shape[0], chunk):
            total += torch.cdist(src[start:start + chunk], dst).min(dim=1).values.sum().item()
        return total / src.shape[0]
    return 0.5 * (one_way(points_a, points_b) + one_way(points_b, points_a))


def cluster(distance, variables, num_groups):
    condensed = squareform(0.5 * (distance + distance.T), checks=False)
    labels = fcluster(linkage(condensed, method="average"), t=num_groups, criterion="maxclust")
    groups = {}
    for name, label in zip(variables, labels):
        groups.setdefault(int(label), []).append(name)
    return sorted(groups.values(), key=lambda g: variables.index(g[0]))


def print_matrix(title, matrix, variables, fmt="{:7.2f}"):
    print(f"\n{title} (row = geometry a, column = target b)")
    print(" " * 9 + "".join(f"{n:>8s}" for n in variables))
    for name, row in zip(variables, matrix):
        print(f"{name:>8s} " + "".join(" " + fmt.format(v) for v in row))


def main():
    parser = ArgumentParser(description="Variable affinity and grouping for geometry experts")
    pp = PipelineParams(parser)
    parser.add_argument("--run_dir", type=str, required=True, help="Output dir of a --groups separate run")
    parser.add_argument("--scans", type=int, default=100, help="Intensity-only refit length in scans")
    parser.add_argument("--subvol_size", type=int, default=128)
    parser.add_argument("--intensity_lr", type=float, default=0.0075)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--num_groups", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--skip_transfer", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)
    pipe = pp.extract(args)

    with open(os.path.join(args.run_dir, "routing.json")) as f:
        run_info = json.load(f)
    # routing maps a variable to [k, c] (older runs) or to a list of [k, c] pairs.
    routing = {name: tuple(kc[0] if isinstance(kc[0], list) else kc)
               for name, kc in run_info["routing"].items()}
    variables = run_info["variables"]
    if any(len(g) != 1 for g in run_info["groups"]):
        raise ValueError("cluster_variables.py needs a run trained with --groups separate")

    config = DATASET_CONFIGS[run_info["dataset_name"]]
    gt_vol, mask, _ = load_multivar_volume(config, run_info["timestep"])
    V, X, Y, Z = gt_vol.shape
    out_dir = os.path.join(args.run_dir, "affinity")
    os.makedirs(out_dir, exist_ok=True)
    timings = {}

    # (b) Chamfer distance between primitive centers, reported in voxel units.
    start = time.time()
    centers = [load_geometry(args.run_dir, routing, n, X, Y, Z).get_xyz.detach() for n in variables]
    voxel = SPATIAL_EXTENT / X
    chamfer_matrix = np.zeros((V, V))
    for a in range(V):
        for b in range(a + 1, V):
            chamfer_matrix[a, b] = chamfer_matrix[b, a] = chamfer(centers[a], centers[b]) / voxel
    timings["chamfer_s"] = time.time() - start
    print_matrix("Chamfer distance [voxels]", chamfer_matrix, variables)
    print("Primitives per variable: " + ", ".join(f"{n}={c.shape[0]}" for n, c in zip(variables, centers)))
    del centers

    results = {"variables": variables, "chamfer": chamfer_matrix.tolist()}
    distances = {"chamfer": chamfer_matrix}

    # (a) Geometry transfer test.
    if not args.skip_transfer:
        psnr_matrix = np.zeros((V, V))
        row_times = []
        for a, name in enumerate(variables):
            row_start = time.time()
            model = load_geometry(args.run_dir, routing, name, X, Y, Z)
            psnr_matrix[a] = transfer_row(model, gt_vol, mask, pipe, args)
            row_times.append(time.time() - row_start)
            print(f"[transfer] geometry={name} ({model.get_xyz.shape[0]} prims) "
                  f"{row_times[-1] / 60:.1f} min: " + "  ".join(f"{n}={p:.2f}" for n, p in zip(variables, psnr_matrix[a])))
            del model
            torch.cuda.empty_cache()
        timings["transfer_s"] = sum(row_times)
        timings["transfer_row_s"] = row_times
        drop = np.diag(psnr_matrix)[None, :] - psnr_matrix
        print_matrix("Transfer PSNR", psnr_matrix, variables)
        print_matrix("Transfer drop [dB] = PSNR(b,b) - PSNR(a,b)", drop, variables)
        results.update({"transfer_psnr": psnr_matrix.tolist(), "transfer_drop": drop.tolist(),
                        "psnr_region": "mask" if mask is not None else "full"})
        # Negative drops (a donor geometry better than b's own) mean high affinity; clip to 0.
        distances["transfer"] = np.clip(drop, 0.0, None)

    for metric, distance in distances.items():
        for k in args.num_groups:
            if k >= V:
                continue
            groups = cluster(distance, variables, k)
            path = os.path.join(out_dir, f"groups_{metric}_K{k}.json")
            with open(path, "w") as f:
                json.dump({"groups": groups, "metric": metric}, f, indent=2)
            print(f"[{metric} K={k}] {groups} -> {path}")

    results["timings"] = timings
    with open(os.path.join(out_dir, "affinity.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTimings: " + ", ".join(f"{k}={v / 60:.1f} min" for k, v in timings.items() if not isinstance(v, list)))


if __name__ == "__main__":
    main()
