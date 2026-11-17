import json
import math
import os
import uuid
from argparse import Namespace

import numpy as np
import skimage
import torch


def convertscantoiter(opt, args, vx, vy, vz, log_path=None):
    s = opt.subvol_size

    nx = math.ceil(vx / s)
    ny = math.ceil(vy / s)
    nz = math.ceil(vz / s)

    iters_per_scan = nx * ny * nz
    
    # 1. 저장을 위한 세트 (중복 방지)
    saving_iterations = set()

    # Backward compatibility for train.py; train_chunk.py passes an explicit
    # per-chunk path after its output directory has been created.
    if log_path is None:
        log_path = os.path.join(args.model_path, "scan_conversion_log.txt")

    # helper: 화면 출력과 파일 기록을 동시에 수행
    def log_and_print(message):
        print(message)
        with open(log_path, 'a') as f:
            f.write(message + "\n")

    # 파일 초기화 (기존 내용 삭제 후 새로 시작)
    with open(log_path, 'w') as f:
        f.write("===== Scan → Iter Conversion =====\n")

    log_and_print(f"volume shape : {(vx, vy, vz)}")
    log_and_print(f"subvol size  : {s}")
    log_and_print(f"nx, ny, nz   : {nx}, {ny}, {nz}")
    log_and_print(f"iters/scan   : {iters_per_scan}\n")

    def conv(name, x):
        new_val = int(round(x * iters_per_scan))
        log_and_print(f"{name:25s} : {x} scans -> {new_val} iters")
        return new_val
    
    # --- 스케줄링 및 파라미터 변환 ---
    opt.iterations = conv("iterations", opt.iterations)
    opt.position_lr_max_steps = conv("position_lr_max_steps", opt.position_lr_max_steps)
    opt.opacity_lr_max_steps = conv("opacity_lr_max_steps", opt.opacity_lr_max_steps)
    opt.scaling_lr_max_steps = conv("scaling_lr_max_steps", opt.scaling_lr_max_steps)
    opt.rotation_lr_max_steps = conv("rotation_lr_max_steps", opt.rotation_lr_max_steps)

    opt.densify_from_iter = conv("densify_from_iter", opt.densify_from_iter)
    opt.densify_until_iter = conv("densify_until_iter", opt.densify_until_iter)
    opt.densification_interval = conv("densification_interval", opt.densification_interval)
    opt.opacity_reset_interval = conv("opacity_reset_interval", opt.opacity_reset_interval)

    # 2. Densification 중요 지점 저장 (시작, 중간, 종료)
    saving_iterations.update([
        opt.densify_from_iter, 
        (opt.densify_from_iter + opt.densify_until_iter) // 2, 
        opt.densify_until_iter
    ])

    log_and_print("darkpruning_iter:")
    opt.darkpruning_iter = [conv("  darkprune", x) for x in opt.darkpruning_iter]
    saving_iterations.update(opt.darkpruning_iter)

    log_and_print("globalpruning_iter:")
    new_global_iters = []
    for x in opt.globalpruning_iter:
        val = conv("  globalprune", x) + 1
        new_global_iters.append(val)
        # 중요: 해당 시점(val)과 바로 전 시점(val - 1) 모두 저장
        saving_iterations.add(val)
        saving_iterations.add(val - 1)
    opt.globalpruning_iter = new_global_iters

    # 3. Warmup 및 초기 구간 저장
    warmup_end = opt.densify_from_iter
    saving_iterations.update([0, warmup_end // 2])

    # 리스트 정렬 및 유효성 검사 (0 ~ total iteration 사이)
    opt.saving_iterations = sorted([
        i for i in saving_iterations if 0 <= i <= opt.iterations
    ])

    log_and_print(f"\n[Generated Saving Iterations]:\n{opt.saving_iterations}")
    log_and_print("==================================")

    # 리턴값 유지
    return iters_per_scan


def configure_scan_schedule(opt, args, subvol_size, volume_shape, log_path):
    """Copy CLI scan values to opt, convert them to iterations, and report them."""
    opt.subvol_size = subvol_size
    opt.init_num = args.init_num
    opt.pruning_remainweight = args.pruning_remainweight

    names = (
        "iterations",
        "position_lr_max_steps",
        "opacity_lr_max_steps",
        "scaling_lr_max_steps",
        "rotation_lr_max_steps",
        "densify_from_iter",
        "densify_until_iter",
        "densification_interval",
        "opacity_reset_interval",
    )
    for name in names:
        setattr(opt, name, getattr(args, name))

    opt.darkpruning_iter = list(args.darkpruning_iter)
    opt.globalpruning_iter = list(args.globalpruning_iter)

    iters_per_scan = convertscantoiter(opt, args, *volume_shape, log_path)
    print("\n=== Scan Schedule Conversion ===")
    print(f"iters_per_scan: {iters_per_scan}")
    print(f"iterations(total): {opt.iterations}")
    print(f"densify_from_iter: {opt.densify_from_iter}")
    print(f"densify_until_iter: {opt.densify_until_iter}")
    print(f"densification_interval: {opt.densification_interval}")
    print(f"opacity_reset_interval: {opt.opacity_reset_interval}")
    print(f"darkpruning_iter: {opt.darkpruning_iter}")
    print(f"globalpruning_iter: {opt.globalpruning_iter}")
    print("===============================\n")
    return iters_per_scan


def load_volume_chunk(config, bounds):
    """Load and normalize only the requested (x, y, z) volume region."""
    path = config["path"]
    volume_shape = (
        config["x_voxel_size"],
        config["y_voxel_size"],
        config["z_voxel_size"],
    )
    transpose = config["transpose"]
    (x_start, y_start, z_start), (x_end, y_end, z_end) = bounds

    print(f"Loading chunk [{x_start}:{x_end}, {y_start}:{y_end}, {z_start}:{z_end}]")
    if config["use_tif"]:
        image = skimage.io.imread(path)
        print(f"Original image shape: {image.shape}")
        if transpose:
            print("transpose volume axes for this dataset")
            image = image.transpose(2, 1, 0)
        subvolume = image[x_start:x_end, y_start:y_end, z_start:z_end]
    else:
        dtype = np.uint16 if "uint16" in path else np.uint8
        mmap_shape = volume_shape[::-1] if transpose else volume_shape
        volume = np.memmap(path, dtype=dtype, mode="r", shape=mmap_shape)
        if transpose:
            volume = volume.transpose(2, 1, 0)
        subvolume = volume[x_start:x_end, y_start:y_end, z_start:z_end].copy()

    divisor = 65535.0 if "uint16" in path else 255.0
    subvolume = subvolume.astype(np.float32) / divisor
    result = torch.from_numpy(subvolume).cuda().detach()
    result.requires_grad = False

    print(f"Chunk min/max: {result.min().item():.6f}/{result.max().item():.6f}")
    memory_gib = result.element_size() * result.nelement() / 1024**3
    print(f"Chunk shape: {result.shape}, memory: {memory_gib:.2f} GB")
    return result


def _coordinate_grid_index(coords, dims):
    """Map each (N, 3) coordinate row to integer (x, y, z) grid indices."""
    columns = []
    counts = []
    for col in range(3):
        _, inverse = np.unique(coords[:, col], return_inverse=True)
        columns.append(inverse.astype(np.int64))
        counts.append(int(inverse.max()) + 1)

    # Coordinate columns may be stored in any axis order; match them by size.
    used = set()
    index = []
    for dim in dims:
        matches = [col for col, count in enumerate(counts) if count == dim and col not in used]
        if not matches:
            raise ValueError(f"Cannot map coordinate columns with sizes {counts} to dims {dims}")
        used.add(matches[0])
        index.append(columns[matches[0]])
    return tuple(index)


def _load_raw_variable(path, raw_size, factor):
    """Read one z-major raw float32 volume as (X, Y, Z), mean-pooled by factor."""
    X, Y, Z = raw_size
    volume = np.fromfile(path, dtype="<f4").reshape(Z, Y, X).transpose(2, 1, 0)
    if factor > 1:
        X, Y, Z = X // factor, Y // factor, Z // factor
        volume = volume[:X * factor, :Y * factor, :Z * factor]
        volume = volume.reshape(X, factor, Y, factor, Z, factor).mean(axis=(1, 3, 5), dtype=np.float64)
    return np.ascontiguousarray(volume, dtype=np.float32)


def load_multivar_volume(config, timestep=None):
    """Load a multivariate volume as a (V, X, Y, Z) CUDA tensor.

    Returns (volume, mask, normalization), where mask is an (X, Y, Z) bool
    tensor or None, and normalization maps each variable to its (min, max)
    in the original units.
    """
    dims = (config["x_voxel_size"], config["y_voxel_size"], config["z_voxel_size"])
    timestep = config["timestep"] if timestep is None else timestep
    if config["format"] == "raw_vars":
        normalization = {}
        volume = np.zeros((len(config["variables"]),) + dims, dtype=np.float32)
        for v, name in enumerate(config["variables"]):
            path = os.path.join(config["data_dir"], config["file_patterns"][name].format(t=timestep))
            channel = _load_raw_variable(path, config["raw_size"], config.get("downsample", 1))
            channel = np.nan_to_num(channel.transpose(config.get("axes", (0, 1, 2))), nan=0.0, posinf=0.0, neginf=0.0)
            vmin, vmax = float(channel.min()), float(channel.max())
            if config["normalize"] == "minmax":
                channel = (channel - vmin) / max(vmax - vmin, 1e-12)
            normalization[name] = (vmin, vmax)
            volume[v] = channel
            print(f"Loading {path}: {name:>8s} raw [{vmin:.6g}, {vmax:.6g}]")
        return torch.from_numpy(volume).cuda(), None, normalization

    data_dir = config["data_dir"]
    variables = config["variables"]

    coords = np.load(os.path.join(data_dir, config["coord_file"]), mmap_mode="r")
    grid_index = _coordinate_grid_index(np.asarray(coords), dims)

    data_path = os.path.join(data_dir, config["data_pattern"].format(t=timestep))
    values = np.asarray(np.load(data_path, mmap_mode="r"), dtype=np.float32)
    values = values.reshape(-1, len(variables))
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Loading {data_path}: {values.shape}")

    normalization = {}
    volume = np.zeros((len(variables),) + dims, dtype=np.float32)
    for v, name in enumerate(variables):
        channel = values[:, v]
        vmin, vmax = float(channel.min()), float(channel.max())
        if config["normalize"] == "minmax":
            # Each variable is normalized independently so no range dominates the loss.
            channel = (channel - vmin) / max(vmax - vmin, 1e-12)
        normalization[name] = (vmin, vmax)
        volume[v][grid_index] = channel
        print(f"  {name:>8s}: raw [{vmin:.6g}, {vmax:.6g}]")

    mask = None
    if config.get("mask_file"):
        mask_values = np.load(os.path.join(data_dir, config["mask_file"])).reshape(-1) > 0
        mask_np = np.zeros(dims, dtype=bool)
        mask_np[grid_index] = mask_values
        mask = torch.from_numpy(mask_np).cuda()
        print(f"  valid mask: {mask_np.mean() * 100:.2f}% of voxels")

    return torch.from_numpy(volume).cuda(), mask, normalization


def make_subvolume_centers(volume_shape, subvolume_size, thin_axis_size=70):
    """Return deterministic patch centers that cover a chunk including its edges."""
    def axis_centers(size, patch_size):
        half = patch_size // 2
        count = math.ceil(size / patch_size)
        return [min(size - half, index * patch_size + half) for index in range(count)]

    x_size, y_size, z_size = volume_shape
    z_patch_size = thin_axis_size if subvolume_size >= z_size else subvolume_size
    centers_x = axis_centers(x_size, subvolume_size)
    centers_y = axis_centers(y_size, subvolume_size)
    centers_z = axis_centers(z_size, z_patch_size)
    return [
        (x, y, z)
        for x in centers_x
        for y in centers_y
        for z in centers_z
    ]


def prepare_output_and_logger(args):
    """Create the run directory and save its parsed command-line arguments."""
    print("args.output_dir", args.output_dir)
    if args.output_dir is not None:
        args.model_path = os.path.join("./output/", args.output_dir)
    elif os.getenv("OAR_JOB_ID"):
        args.model_path = os.path.join("./output/", os.getenv("OAR_JOB_ID")[:10])
    else:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log:
        cfg_log.write(str(Namespace(**vars(args))))


def write_volume_json(model_path, shape):
    """Write volume.json so the viewer can clip ray samples to the volume.

    shape is (X, Y, Z) in Gaussian x, y, z order. Training maps voxel centers of
    every axis onto the x axis grid (step 2/X, centered), so the box is those
    centers padded by half a voxel.
    """
    X = shape[0]
    step = 2.0 / X
    bbox_min, bbox_max = [], []
    for n in shape:
        first = -1.0 + ((X - n) // 2) * step
        bbox_min.append(first - step / 2)
        bbox_max.append(first + (n - 1) * step + step / 2)
    info = {"volume_shape": [int(n) for n in shape], "bbox_min": bbox_min, "bbox_max": bbox_max}
    with open(os.path.join(model_path, "volume.json"), "w") as f:
        json.dump(info, f, indent=2)


def write_training_parameters(
    log_path,
    args,
    opt,
    chunk_idx,
    chunk_coords,
    chunk_size,
    offset,
):
    """Initialize the PSNR log with CLI and derived training parameters."""
    cli_param_names = (
        "chunk_idx", "dataset_name", "output_dir", "subvol_size", "iterations",
        "densify_from_iter", "densify_until_iter", "densification_interval",
        "opacity_reset_interval", "darkpruning_iter", "globalpruning_iter",
        "position_lr_max_steps", "opacity_lr_max_steps", "scaling_lr_max_steps",
        "rotation_lr_max_steps", "position_lr_init", "position_lr_final",
        "opacity_lr_init", "opacity_lr_final", "scaling_lr_init",
        "scaling_lr_final", "rotation_lr_init", "rotation_lr_final",
        "position_lr_delay_mult", "lambda_dssim", "densify_grad_threshold",
        "percent_dense", "init_num", "pruning_remainweight", "batch_size",
        "sync_interval",
    )

    with open(log_path, "w") as log:
        script_name = getattr(args, "training_script", "train_chunk.sh")
        log.write(f"=== Training Parameters ({script_name}) ===\n")
        for key in cli_param_names:
            value = getattr(args, key, "N/A")
            if ("lr_init" in key or "lr_final" in key) and isinstance(value, (int, float)):
                log.write(f"{key}: {value:.6f}\n")
            elif "threshold" in key and isinstance(value, (int, float)):
                log.write(f"{key}: {value:.10f}\n")
            else:
                log.write(f"{key}: {value}\n")

        log.write("\n=== Training Parameters (converted) ===\n")
        log.write(f"densify_grad_threshold: {opt.densify_grad_threshold:.10f}\n")
        log.write(f"percent_dense: {opt.percent_dense:.6f}\n")
        log.write(f"scaling_lr_init: {opt.scaling_lr_init:.10f}\n")
        log.write(f"scaling_lr_final: {opt.scaling_lr_final:.10f}\n")
        log.write(f"iterations: {opt.iterations}\n")
        log.write(f"densify_from_iter: {opt.densify_from_iter}\n")
        log.write(f"densify_until_iter: {opt.densify_until_iter}\n")
        log.write(f"densification_interval: {opt.densification_interval}\n")
        log.write(f"subvol_size: {opt.subvol_size}\n")
        log.write(f"\n=== Training chunk {chunk_idx}/{len(chunk_coords)-1} ===\n")
        log.write(f"Chunk coordinates: {chunk_coords[chunk_idx]}\n")
        log.write(f"Chunk size: {chunk_size}\n")
        log.write(f"Chunk offset: {offset.detach().cpu().numpy()}\n")
