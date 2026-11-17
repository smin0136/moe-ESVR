"""Voxelize trained multivariate (MoE) runs and save the volumes as TIFs.

Usage: python export_multivar_tif.py --out output/budget_compare/<dataset> \
           --run label=output/multivar/<dataset>/<run> [--run ...]

For every run (a train_multivar.py output: expert_k/point_cloud.ply + routing.json)
and every variable it writes into <out>/<label>/:
  pred_<var>.tif          the reconstruction (clamped to [0, 1], as evaluated)
  err_<var>.tif           |pred - GT|
  expert<k>_<var>.tif     contribution of expert k alone (unclamped), only for
                          variables routed to more than one expert
and the ground truth once into <out>/gt/gt_<var>.tif. Volumes are float32 in
ZYX order (ImageJ / Fiji stacks over z). A summary goes to <out>/index.tsv.
"""
import json
import math
import os
from argparse import ArgumentParser

import numpy as np
import tifffile
import torch

from arguments import PipelineParams
from dataset_configs import DATASET_CONFIGS
from gaussian_renderer import query_experts
from scene import GaussianModel
from utils.train_utils import load_multivar_volume

SPATIAL_EXTENT = 2.0


def save(path, volume):
    tifffile.imwrite(path, volume.permute(2, 1, 0).contiguous().cpu().numpy().astype(np.float32), imagej=True)


def members_of(entry):
    # Separate models store one [expert, channel] pair, routed models a list of pairs.
    return [tuple(entry)] if isinstance(entry[0], int) else [tuple(pair) for pair in entry]


@torch.no_grad()
def export_run(label, run_dir, out_dir, pipe, gt_cache):
    info = json.load(open(os.path.join(run_dir, "routing.json")))
    config = DATASET_CONFIGS[info["dataset_name"]]
    variables = info["variables"]
    key = (info["dataset_name"], info["timestep"])
    if key not in gt_cache:
        gt_cache.clear()
        gt_cache[key] = load_multivar_volume(config, info["timestep"])[:2]
    gt_vol, mask = gt_cache[key]
    _, X, Y, Z = gt_vol.shape
    sVoxel = [SPATIAL_EXTENT, Y / X * SPATIAL_EXTENT, Z / X * SPATIAL_EXTENT]

    gt_dir = os.path.join(out_dir, "gt")
    if not os.path.isdir(gt_dir):
        os.makedirs(gt_dir)
        for v, name in enumerate(variables):
            save(os.path.join(gt_dir, f"gt_{name}.tif"), gt_vol[v])

    routing = {name: members_of(info["routing"][name]) for name in variables}
    experts = {}
    for k in sorted({k for pairs in routing.values() for k, _ in pairs}):
        expert = GaussianModel(X, Y, Z)
        expert.load_ply(os.path.join(run_dir, f"expert_{k}", "point_cloud.ply"))
        experts[k] = expert

    run_out = os.path.join(out_dir, label)
    os.makedirs(run_out, exist_ok=True)
    psnr, mse_all = {}, []
    for v, name in enumerate(variables):
        pairs = routing[name]
        pred = query_experts([experts[k] for k, _ in pairs], [c for _, c in pairs],
                             [0, 0, 0], [X, Y, Z], sVoxel, pipe)["vol"]
        pred = torch.clamp(pred.reshape(X, Y, Z), 0.0, 1.0)
        err = (pred - gt_vol[v]).abs()
        mse = (err ** 2).mean().item()
        mse_all.append(mse)
        psnr[name] = -10.0 * math.log10(mse)
        save(os.path.join(run_out, f"pred_{name}.tif"), pred)
        save(os.path.join(run_out, f"err_{name}.tif"), err)
        if len(pairs) > 1:
            for k, c in pairs:
                part = query_experts([experts[k]], [c], [0, 0, 0], [X, Y, Z], sVoxel, pipe)["vol"]
                save(os.path.join(run_out, f"expert{k}_{name}.tif"), part.reshape(X, Y, Z))

    metrics = json.load(open(os.path.join(run_dir, "metrics.json")))
    overall = -10.0 * math.log10(sum(mse_all) / len(mse_all))
    print(f"{label}: overall {overall:.2f} dB (metrics.json {metrics['overall_psnr']:.2f}), "
          f"{metrics['size_mb']:.2f} MB")
    usage = " ".join(f"{n}:{''.join(str(k) for k, _ in routing[n])}" for n in variables)
    return [label, run_dir, f"{overall:.2f}", f"{metrics['size_mb']:.2f}",
            str(sum(metrics["num_primitives"])), usage] + [f"{psnr[n]:.2f}" for n in variables]


def main():
    parser = ArgumentParser(description=__doc__)
    pp = PipelineParams(parser)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run", action="append", required=True, help="label=run_dir")
    args = parser.parse_args()
    pipe = pp.extract(args)

    os.makedirs(args.out, exist_ok=True)
    rows, gt_cache, variables = [], {}, None
    for spec in args.run:
        label, run_dir = spec.split("=", 1)
        variables = json.load(open(os.path.join(run_dir, "routing.json")))["variables"]
        rows.append(export_run(label, run_dir, args.out, pipe, gt_cache))
    with open(os.path.join(args.out, "index.tsv"), "w") as f:
        f.write("\t".join(["label", "run_dir", "overall_psnr", "size_mb", "N", "usage"] + variables) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


if __name__ == "__main__":
    main()
