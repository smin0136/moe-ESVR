"""Multivariate ESVR with spatial x variable routing (attribute MoE).

One primitive bank. Each primitive has a shared center, G shape experts
(scale, rotation) and one intensity per variable, and it routes every variable
to one of its shapes:

  --groups g.json   G = number of groups; variable v starts routed to its
                    group's shape with probability --home_prob
  --groups shared   G = 1, i.e. the fully shared hard baseline

Schedule: soft routing until --lasso_scans[0]; then the routing temperature
is annealed from 1 to --tau_min while a group-lasso penalty over
(primitive, shape) removes unused shapes; at --lasso_scans[1] routing becomes
hard (argmax) and training continues. Size counts only the shapes and
intensities that hard routing still uses.

Variable routing experiments stay in train_multivar.py.
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
from gaussian_renderer import query_attr_moe
from prune import compute_fa, prune_list, prune_zeros
from scene.attr_moe_model import INTENSITY_MIN, AttrMoEGaussianModel
from train_multivar import SPATIAL_EXTENT, SPATIAL_LR_SCALE, axis_centers, psnr_from_mse, resolve_groups
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.train_utils import configure_scan_schedule, load_multivar_volume, prepare_output_and_logger


def temperature(iteration, start, end, tau_min):
    """Log-linear anneal from 1 at start to tau_min at end."""
    if iteration <= start:
        return 1.0
    progress = min(1.0, (iteration - start) / max(1, end - start))
    return math.exp(progress * math.log(tau_min))


@torch.no_grad()
def routing_stats(model, variables, hard, tau):
    """Per variable: share of its active primitives routed to each shape."""
    if hard:
        choice = model.hard_route()
    else:
        choice = torch.softmax(model._route / tau, dim=-1).argmax(dim=-1)
    active = model.variable_intensity(choice) >= INTENSITY_MIN  # [N, V]
    stats = {}
    for v, name in enumerate(variables):
        picked = choice[active[:, v], v]
        counts = torch.bincount(picked, minlength=model.num_shapes).float()
        stats[name] = {"active": int(active[:, v].sum()),
                       "shape_share": (counts / counts.sum().clamp(min=1)).tolist()}
    return stats


@torch.no_grad()
def evaluate(model, variables, gt_vol, mask, pipe, hard, tau):
    X, Y, Z = gt_vol.shape[1:]
    full_sVoxel = [SPATIAL_EXTENT, Y / X * SPATIAL_EXTENT, Z / X * SPATIAL_EXTENT]
    result = {"psnr": {}, "psnr_mask": {}}
    mse_all, mse_mask_all = [], []
    for v, name in enumerate(variables):
        pred = query_attr_moe(model, v, [0, 0, 0], [X, Y, Z], full_sVoxel, pipe, tau=tau, hard=hard)["vol"]
        pred = torch.clamp(pred.reshape(X, Y, Z), 0.0, 1.0)
        sq_err = (pred - gt_vol[v]) ** 2
        mse = sq_err.mean().item()
        mse_all.append(mse)
        result["psnr"][name] = psnr_from_mse(mse)
        if mask is not None:
            mse_mask = sq_err[mask].mean().item()
            mse_mask_all.append(mse_mask)
            result["psnr_mask"][name] = psnr_from_mse(mse_mask)
    result["overall_psnr"] = psnr_from_mse(sum(mse_all) / len(mse_all))
    result["mean_psnr"] = sum(result["psnr"].values()) / len(variables)
    if mask is not None:
        result["overall_psnr_mask"] = psnr_from_mse(sum(mse_mask_all) / len(mse_mask_all))
    storage = model.storage(hard)
    result["num_primitives"] = [model.get_xyz.shape[0]]
    result["size_mb"] = storage["bytes"] / 1024**2
    result["dense_size_mb"] = storage["dense_bytes"] / 1024**2
    result["active_intensity_frac"] = storage["active_intensity_frac"]
    result["shapes_per_primitive"] = storage["shapes_per_primitive"]
    result["hard"] = hard
    result["tau"] = tau
    result["routing_stats"] = routing_stats(model, variables, hard, tau)
    return result


def format_eval(result, variables):
    lines = ["  PSNR      : " + "  ".join(f"{n}={result['psnr'][n]:.2f}" for n in variables)]
    if result["psnr_mask"]:
        lines.append("  PSNR(mask): " + "  ".join(f"{n}={result['psnr_mask'][n]:.2f}" for n in variables))
    overall = f"  overall={result['overall_psnr']:.2f} dB  mean={result['mean_psnr']:.2f} dB"
    if "overall_psnr_mask" in result:
        overall += f"  overall(mask)={result['overall_psnr_mask']:.2f} dB"
    lines.append(overall)
    lines.append(f"  primitives={result['num_primitives'][0]}  size={result['size_mb']:.3f} MB "
                 f"(dense {result['dense_size_mb']:.3f} MB)  shapes/prim={result['shapes_per_primitive']:.2f}  "
                 f"active intensities={result['active_intensity_frac'] * 100:.1f}%  hard={result['hard']}  tau={result['tau']:.3f}")
    lines.append("  shape share per variable (active primitives):")
    for name in variables:
        s = result["routing_stats"][name]
        lines.append(f"    {name:>8s}: " + " ".join(f"S{g}={p * 100:5.1f}%" for g, p in enumerate(s["shape_share"]))
                     + f"  (active {s['active']})")
    return "\n".join(lines)


def training(dataset, opt, pipe, args):
    config = DATASET_CONFIGS[args.dataset_name]
    variables = config["variables"]
    groups = resolve_groups(args.groups, variables)
    G, V = len(groups), len(variables)
    home = [next(g for g, group in enumerate(groups) if name in group) for name in variables]
    if pipe.compute_cov3D_python:
        raise ValueError("train_spatial.py needs compute_cov3D_python=False")

    prepare_output_and_logger(dataset)
    model_path = dataset.model_path
    gt_vol, mask, normalization = load_multivar_volume(config, args.timestep)
    _, X, Y, Z = gt_vol.shape
    print(f"Variables: {variables}")
    expert_attrs = tuple(a.strip() for a in args.expert_attrs.split(","))
    print(f"Experts ({G}, attrs={expert_attrs}): {groups}  home={home}  lasso={args.lasso_lambda}  tau_min={args.tau_min}")

    iters_per_scan = configure_scan_schedule(
        opt, args, args.subvol_size, (X, Y, Z), os.path.join(model_path, "scan_conversion_log.txt")
    )
    eval_iters = {s * iters_per_scan for s in args.eval_scans}
    lasso_start, hard_iter = (s * iters_per_scan for s in args.lasso_scans)

    run_info = {"dataset_name": args.dataset_name, "variables": variables, "groups": groups, "home": home,
                "timestep": args.timestep if args.timestep is not None else config["timestep"],
                "normalization": normalization, "args": vars(args)}
    with open(os.path.join(model_path, "routing.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    patch = [min(args.subvol_size, d) for d in (X, Y, Z)]
    half = [p // 2 for p in patch]
    subvol_centers = [(cx, cy, cz) for cx in axis_centers(X, patch[0])
                      for cy in axis_centers(Y, patch[1]) for cz in axis_centers(Z, patch[2])]
    x_normalized = torch.linspace(-SPATIAL_EXTENT / 2, SPATIAL_EXTENT / 2 - 2 / X, steps=X)
    y_normalized = x_normalized[(X - Y) // 2:(X - Y) // 2 + Y]
    z_normalized = x_normalized[(X - Z) // 2:(X - Z) // 2 + Z]
    tv_vol_nVoxel = torch.tensor(patch)
    tv_vol_sVoxel = torch.tensor(SPATIAL_EXTENT / X) * tv_vol_nVoxel
    print(f"Patch {patch}, subvolumes per scan: {len(subvol_centers)}")

    model = AttrMoEGaussianModel(X, Y, Z, num_variables=V, num_shapes=G, expert_attrs=expert_attrs)
    model.create_random_onthefly(opt.init_num, args.spatial_lr_scale, SPATIAL_EXTENT, seed=42 + args.seed,
                                 knn_on_scaled=args.init_knn_scaled, home=home, home_prob=args.home_prob)
    model.training_setup(opt, route_lr=args.route_lr)
    weight = 1.0 / V

    log_path = os.path.join(model_path, "psnr_log.tsv")
    with open(log_path, "w") as f:
        header = ["iteration", "scan", "loss"] + [f"psnr_{n}" for n in variables] + ["overall_psnr", "mean_psnr"]
        if mask is not None:
            header += [f"psnr_mask_{n}" for n in variables] + ["overall_psnr_mask"]
        header += ["num_primitives", "size_mb", "dense_size_mb", "shapes_per_primitive", "hard", "elapsed_s"]
        f.write("\t".join(header) + "\n")

    def log_eval(iteration, loss_value, elapsed, hard, tau):
        result = evaluate(model, variables, gt_vol, mask, pipe, hard, tau)
        print(f"\n[ITER {iteration} | scan {iteration // iters_per_scan}] loss={loss_value:.6f} elapsed={elapsed / 60:.1f} min")
        print(format_eval(result, variables))
        row = [iteration, iteration // iters_per_scan, f"{loss_value:.6f}"]
        row += [f"{result['psnr'][n]:.4f}" for n in variables]
        row += [f"{result['overall_psnr']:.4f}", f"{result['mean_psnr']:.4f}"]
        if mask is not None:
            row += [f"{result['psnr_mask'][n]:.4f}" for n in variables] + [f"{result['overall_psnr_mask']:.4f}"]
        row += [result["num_primitives"][0], f"{result['size_mb']:.4f}", f"{result['dense_size_mb']:.4f}",
                f"{result['shapes_per_primitive']:.3f}", int(hard), f"{elapsed:.1f}"]
        with open(log_path, "a") as f:
            f.write("\t".join(map(str, row)) + "\n")
        return result

    start_time = time.time()
    subvol_ptr = 0
    ema_loss = 0.0
    result = None
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):
        scan_aligned_iter = (iteration // iters_per_scan) * iters_per_scan
        model.update_learning_rate(scan_aligned_iter)
        if iteration == 1:
            model.prune_out_bbox(SPATIAL_EXTENT)
        hard = G > 1 and iteration >= hard_iter
        tau = temperature(iteration, lasso_start, hard_iter, args.tau_min)

        cx, cy, cz = subvol_centers[subvol_ptr]
        subvol_ptr = (subvol_ptr + 1) % len(subvol_centers)
        tv_vol_center = (x_normalized[cx], y_normalized[cy], z_normalized[cz])
        gt_subvol = gt_vol[:, cx - half[0]:cx + half[0], cy - half[1]:cy + half[1], cz - half[2]:cz + half[2]]

        # One backward per variable; densification uses the largest per-variable
        # xyz gradient, as with densify_stat=max in train_multivar.py.
        N = model.get_xyz.shape[0]
        grad_stat = torch.zeros((N, 1), device="cuda")
        visibility = torch.zeros(N, dtype=torch.bool, device="cuda")
        radii_max = torch.zeros(N, device="cuda")
        total_loss = 0.0
        for v in range(V):
            pkg = query_attr_moe(model, v, tv_vol_center, tv_vol_nVoxel, tv_vol_sVoxel, pipe, tau=tau, hard=hard)
            vol_pred = pkg["vol"].reshape(gt_subvol.shape[1:])
            gt = gt_subvol[v]
            loss_v = (1.0 - opt.lambda_dssim) * l1_loss(vol_pred, gt) + opt.lambda_dssim * (1.0 - ssim(vol_pred, gt))
            total_loss += weight * loss_v.item()
            previous = model._xyz.grad.clone() if model._xyz.grad is not None else None
            (weight * loss_v).backward()
            grad = model._xyz.grad if previous is None else model._xyz.grad - previous
            grad_stat = torch.maximum(grad_stat, torch.norm(grad[:, :3], dim=-1, keepdim=True) / weight)
            visibility |= pkg["visibility_filter"]
            radii_max = torch.maximum(radii_max, pkg["radii"].float())

        if args.lasso_lambda > 0 and lasso_start <= iteration < hard_iter:
            (args.lasso_lambda * model.shape_usage(tau).sum(dim=1).mean()).backward()

        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                model.max_radii2D[visibility] = torch.max(model.max_radii2D[visibility], radii_max[visibility])
                model.add_densification_stats_norm(grad_stat, visibility)
            if iteration % 100 == 0:
                torch.cuda.empty_cache()

            if not args.no_pruning and (iteration in opt.darkpruning_iter or iteration == opt.iterations):
                prune_zeros(model, None, pipe, None, X, Y, Z, SPATIAL_EXTENT)
                model.prune_dark_gaussians()
            if iteration in opt.globalpruning_iter and not args.no_pruning:
                prune_zeros(model, None, pipe, None, X, Y, Z, SPATIAL_EXTENT)
                model.prune_dark_gaussians()
                _, imp_list, _ = prune_list(model, None, pipe, None, X, Y, Z, SPATIAL_EXTENT)
                model.prune_gaussians(opt.pruning_remainweight, imp_list * compute_fa(model.get_scaling))

            below_cap = args.max_primitives <= 0 or model.get_xyz.shape[0] < args.max_primitives
            if opt.densify_from_iter <= iteration <= opt.densify_until_iter and below_cap:
                if iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    model.densify_and_prune(opt.densify_grad_threshold, 1 / 255, SPATIAL_EXTENT,
                                            size_threshold, None, iteration, SPATIAL_EXTENT)
                    count = model.get_xyz.shape[0]
                    if args.max_primitives > 0 and count > args.max_primitives:
                        model.prune_gaussians((count - args.max_primitives) / count, model.get_aggregate_intensity)
                if iteration % opt.opacity_reset_interval == 0:
                    model.reset_opacity()

            if iteration < opt.iterations:
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)

            if G > 1 and iteration == hard_iter:
                print(f"\n[ITER {iteration}] routing switched to hard")

            ema_loss = 0.4 * total_loss + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.7f}", "N": model.get_xyz.shape[0], "tau": f"{tau:.3f}"})
                progress_bar.update(10)
            if iteration in eval_iters or iteration == opt.iterations:
                result = log_eval(iteration, ema_loss, time.time() - start_time, hard, tau)

    progress_bar.close()
    total_time = time.time() - start_time
    final_hard = G > 1 and opt.iterations >= hard_iter
    model.save_npz(os.path.join(model_path, "attr_moe.npz"), final_hard)
    result["train_time_s"] = total_time
    result["groups"] = groups
    result["routing_mode"] = "spatial"
    with open(os.path.join(model_path, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Final] {args.dataset_name} groups={args.groups} spatial routing time={total_time / 60:.1f} min")
    print(format_eval(result, variables))


if __name__ == "__main__":
    parser = ArgumentParser(description="Multivariate ESVR with spatial x variable routing")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    multivar_datasets = sorted(n for n, c in DATASET_CONFIGS.items() if c.get("format") in ("npy_points", "raw_vars"))
    parser.add_argument("--dataset_name", type=str, required=True, choices=multivar_datasets)
    parser.add_argument("--groups", type=str, default="shared", help="shared | path to groups json (sets G and the home shapes)")
    parser.add_argument("--expert_attrs", type=str, default="scale,rot",
                        help="Attributes held per expert slot: comma-separated subset of scale,rot,intensity")
    parser.add_argument("--home_prob", type=float, default=0.9, help="Initial routing probability of the home shape")
    parser.add_argument("--route_lr", type=float, default=0.01)
    parser.add_argument("--lasso_lambda", type=float, default=0.0, help="Group-lasso weight over (primitive, shape)")
    parser.add_argument("--lasso_scans", nargs=2, type=int, default=[400, 2000],
                        help="Anneal + penalty window in scans; routing becomes hard at its end")
    parser.add_argument("--tau_min", type=float, default=0.05, help="Routing temperature at the end of the window")
    parser.add_argument("--timestep", type=int, default=None, help="Overrides the config timestep")
    parser.add_argument("--subvol_size", type=int, default=128)
    parser.add_argument("--init_num", type=int, default=100_000, help="Initial primitives")
    parser.add_argument("--pruning_remainweight", type=float, default=0.66)
    parser.add_argument("--darkpruning_iter", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--globalpruning_iter", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--eval_scans", nargs="+", type=int, default=[400, 1200, 2000])
    parser.add_argument("--init_knn_scaled", action="store_true")
    parser.add_argument("--max_primitives", type=int, default=0, help="Primitive cap (0 = off)")
    parser.add_argument("--spatial_lr_scale", type=float, default=SPATIAL_LR_SCALE)
    parser.add_argument("--no_pruning", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    safe_state(args.quiet)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)
    print("\nTraining complete.")
