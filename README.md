# ESVR: 3D Ellipsoid-based Sparse Volume Rendering via Structure-aware Primitive Learning and Per-primitive Ray Sampling

![ESVR overview](./fig/teaser.png)

Suemin Jeon*, Youjin Kim*, Jungwoo Park, Kyungryun Lee, Won-Ki Jeong

\* Equal contribution

## Description

This repository contains the implementation of **ESVR: 3D Ellipsoid-based
Sparse Volume Rendering via Structure-aware Primitive Learning and
Per-primitive Ray Sampling**.

## Setup

Docker is recommended for a reproducible CUDA environment. 

### Tested Training Environment

Training and inference were tested with:

- **PyTorch:** `1.13.1+cu117`
- **NVCC:** `11.7`

### Installation

Clone the repository together with its submodules:

```bash
git clone --recursive https://github.com/hvcl/ESVR.git
cd ESVR
```

Build the Docker image and start an interactive container:

```bash
./docker/build.sh
./docker/use_docker.sh
```

To select a GPU and assign a container name, pass them as arguments:

```bash
./docker/use_docker.sh 0 esvr-container
```

Inside the container, install the Python dependencies and CUDA extensions:

```bash
export HOME=/tmp/esvr-home
mkdir -p "$HOME"

python -m pip install --user -r requirements.txt
python -m pip install --user ./submodules/simple-knn
python -m pip install --user ./submodules/gaussian-rasterization-voxelization
```

## Dataset

Dataset paths, volume dimensions, chunk sizes, and training modes are defined
in `dataset_configs.py`. An example Aneurysm volume is provided in `./data`.

Add a new entry to `DATASET_CONFIGS` before training on a custom volume.

## Training

### Single-volume and Sequential Chunk Training

Use `train_chunk.sh` for datasets configured with `training_mode: "full"` or
`training_mode: "chunk_sequential"`. The script trains all configured chunks
in sequence.

```bash
./train_chunk.sh aneurysm
```

If no dataset name is provided, the default dataset defined in the script is
used.

### Parallel Chunk Training

Use `train_parallel.sh` for datasets configured with
`training_mode: "chunk_parallel"`. Multiple chunks are assigned to the
available GPUs and trained in parallel.

```bash
./train_parallel.sh hemibrain_lLN2P
```

Set `BATCH_SIZE` in `train_parallel.sh` to control the number of chunks trained
at the same time. It should not exceed the number of available GPUs.

## Multivariate MoE (Channel Routing)

`train_multivar.py` extends ESVR to multivariate volumes (V variables on the
same grid) with a mixture of geometry experts. The final method is **channel
routing with a cost-weighted group lasso**:

- There are K experts. Each expert is one Gaussian set (shared position,
  scale and rotation) with one intensity channel per variable. A variable is
  the sum of the experts it is routed to, each contributing that variable's
  channel.
- Training starts with every channel on. Channels outside the expert's initial
  group start at a low intensity (`--off_init 0.02`).
- From scan 400 to 2000 (`--lasso_scans`) a group lasso on each channel's
  intensity RMS, weighted by the expert's size, pushes unused channels to zero.
- At scan 2000 every channel whose RMS is below `--channel_threshold` is
  switched off, which fixes the routing. Every variable keeps at least its
  strongest expert.
- Densification, dark pruning and the scan-1200 global prune follow ESVR.
  Densification uses the largest per-variable xyz gradient
  (`--densify_stat max`), so variables do not cancel each other.

### Running

```bash
./train_multivar.sh <dataset_name> <groups> [train_multivar.py args]
```

`<groups>` sets the experts and their initial variables:

- `shared`: one expert holding every variable.
- `separate`: one expert per variable (K = V). This needs no preprocessing and
  is the recommended start for a new dataset.
- A path to a groups json, `{"groups": [["a"], ["b", "c"], ...]}`, gives one
  expert per group.

Transfer groups are made from a `separate` run. `cluster_variables.py` refits
each variable on every other variable's geometry and clusters the variables by
the PSNR drop:

```bash
./train_multivar.sh rocom_masked separate --output_dir multivar/rocom_masked/separate
python cluster_variables.py --run_dir output/multivar/rocom_masked/separate
# writes .../separate/affinity/groups_transfer_K{2,3,4}.json
```

Best settings per dataset (overall PSNR over all variables, final size). The
groups used for ROCOM and SNU are in `configs/groups/`.

```bash
NEW="--max_primitives 500000 --split_budget --init_num 100000"
CH="--routing channel --off_init 0.02"

# Ionization, 46.83 dB / 6.79 MB
./train_multivar.sh ionization_ds2 separate $NEW $CH --lasso_lambda 1e-1
# Combustion, 46.08 dB / 4.33 MB
./train_multivar.sh combustion_ds2 separate $NEW $CH --lasso_lambda 3e-1
# ROCOM (masked, t=2), 54.59 dB / 11.57 MB
./train_multivar.sh rocom_masked configs/groups/rocom_masked_transfer_K4.json $CH --lasso_lambda 1e-2
# SNU (down_down, t=0), 34.90 dB / 15.84 MB
./train_multivar.sh snu_down_down configs/groups/snu_down_down_transfer_K4.json \
    --subvol_size 32 --iterations 1000 --position_lr_max_steps 800 \
    --scaling_lr_max_steps 800 --rotation_lr_max_steps 800 \
    --densification_interval 16 --init_num 30000 --max_primitives 250000 \
    --split_budget --eval_scans 100 400 800 --timestep 0 --lasso_scans 400 800 \
    $CH --lasso_lambda 1e-2
```

- The output goes to `output/multivar/<dataset>/<groups name>` unless you pass
  `--output_dir multivar/<dataset>/<run>`.
- The datasets are defined in `dataset_configs.py` with `format: "npy_points"`.
  Their data directories are on the lab server.
- Short runs need `--lasso_scans` inside `--iterations`. Otherwise the routing
  is never fixed.

These numbers were obtained before 2026-10-29. Since then, an expert that has
reached its primitive cap still removes centers outside the volume box during
densification, as ESVR does. Add `--no_cap_bbox_prune` to reproduce the old
behavior exactly.

### Outputs

| File | Content |
|---|---|
| `expert_<k>/point_cloud.ply` | Expert k. Its intensities are stored as `intensity_<i>`, one per channel. |
| `routing.json` | Variables, initial groups, final routing table, the active mask `active[k][v]`, normalization, and all arguments. |
| `metrics.json` | Final per-variable and overall PSNR (and masked PSNR for ROCOM), primitives per expert, and size in MB. |
| `psnr_log.tsv` | The same values at every `--eval_scans` scan. |
| `volume.json` | `volume_shape` (X, Y, Z) and `bbox_min` / `bbox_max` of the volume in Gaussian coordinates, for the viewer. |
| `scan_conversion_log.txt` | The scan → iteration schedule. |

- `python export_multivar_tif.py --out <dir> --run <label>=<run dir>` writes
  each variable's prediction, its error and the per-expert contributions as
  TIFs.
- `logs/write_volume_json.py` adds `volume.json` to runs made before it
  existed.

### Changes to Shared Code

- `scene/gaussian_model.py`
  - `GaussianModel(..., num_channels)` holds several intensity channels.
  - Pruning uses `get_aggregate_intensity`, the max over channels, so a
    primitive stays while any variable needs it.
  - `save_ply` / `load_ply` read and write `intensity_<i>`. Single-channel
    files are unchanged.
  - `prune_out_bbox` takes `margin` and `verbose` arguments.
- `gaussian_renderer/__init__.py`
  - `query_experts` voxelizes several experts as one set, each with its own
    channel.
- `utils/train_utils.py`
  - `load_multivar_volume`, `configure_scan_schedule` and `write_volume_json`.
- `prune.py`
  - Importance scores use the aggregate intensity.
- `dataset_configs.py`
  - Adds `rocom_masked`, `snu_down_down`, `ionization_ds2` and
    `combustion_ds2`.
- Viewer
  - Opens MoE runs and renders one variable at a time (see *Viewer*).

### Experimental Options

`--routing` defaults to `hard`, which keeps the groups fixed with no channel
routing, so the final method always needs `--routing channel`. The options
below were used for ablations. They are all off by default.

| Option | Purpose |
|---|---|
| `--topk N --topk_mode exact\|max` | At most N experts per variable. |
| `--budget_mode exact\|usage\|ratio\|pool --target_mb M` | Prune to an exact total size, with equal, usage-weighted, fixed-ratio (`--budget_ratio`) or global shares. |
| `--expert_target N...` | Exact primitives per expert. |
| `--fixed_routing <metrics.json>` | Retrain with an earlier run's final routing fixed from the start. |
| `--uniform_init K` | K identical experts with every channel on, and no prior groups. |
| `--scale_bounds min:max ...` | Per-expert scale range in voxels (coarse / mid / fine experts). |
| `--progressive_scans s...` | Start scan per expert. Later experts learn the residual. |

`train_spatial.py` and `scene/attr_moe_model.py` are a spatial-routing variant
(experts that share attributes per primitive). It was below channel routing at
equal size and is kept for reference.

`logs/make_*_jobs.sh` regenerate the job lists of the ablations, and
`logs/summarize_*.py` collect their results.

## Evaluation

Use `inference.py` to reconstruct a volume from a trained PLY file and evaluate
it against the ground-truth volume specified in `dataset_configs.py`.

Evaluate a trained point cloud and print the full-volume PSNR:

```bash
python inference.py \
  --dataset_name aneurysm \
  --input_ply ./output/point_cloud.ply
```

Add `--save_tif` to save the reconstructed volume as a TIFF file:

```bash
python inference.py \
  --dataset_name aneurysm \
  --input_ply ./output/point_cloud.ply \
  --save_tif \
  --output_tif ./output/aneurysm.tif \
  --output_dtype uint8 \
  --chunk_size 512
```

<details>
<summary>Command-line arguments for <code>inference.py</code></summary>

- `--dataset_name`: Dataset key defined in `dataset_configs.py`. Its volume
  dimensions and ground-truth path are used for reconstruction and evaluation.
- `--input_ply`: Path to the trained Gaussian point-cloud PLY file.
- `--save_tif`: Save the reconstructed volume as a TIFF file. If omitted, only
  evaluation is performed.
- `--output_tif`: Output TIFF path (default: `./output/volume_merged.tif`).
- `--output_dtype`: Output voxel type, either `uint8` or `uint16` (default:
  `uint8`).
- `--chunk_size`: Number of voxels queried along each axis per chunk (default:
  `1024`). Reduce this value if inference runs out of GPU memory.
- `--full`: Query the entire volume at once. This can be faster for small
  volumes but requires more GPU memory.

</details>

## Viewer

![ESVR Viewer](./fig/viewer.png)

### Tested Build Environment

The Viewer was tested with:

- **Operating system:** Windows 10 (64-bit)
- **Visual Studio:** Visual Studio 2019 Community
- **MSVC compiler:** 19.29.30159 (x64)
- **MSVC toolset:** 14.29.30133 (v142)
- **Windows SDK:** 10.0.19041.0 / 10.0.26100.0
- **CMake:** 4.3.0
- **CUDA Toolkit / NVCC:** 11.8 / 11.8.89
- **NVIDIA driver:** 571.96
- **GPU:** NVIDIA GeForce RTX 3090 (24 GB)
- **Conda:** 25.11.1
- **Build configuration:** `RelWithDebInfo` (x64)

### Interactive GUI

A pre-built Viewer executable is included in `SIBR_viewer/install/bin`, so you
can run the Viewer directly without building it from source.

Launch the Viewer from the `SIBR_viewer/install/bin` directory and
specify the model output directory with `-m`:

```bash
cd SIBR_viewer/install/bin
SIBR_gaussianViewer_app_rwdi.exe -m <output_path>
```

MoE models (`train_multivar.py` outputs with `expert_k/point_cloud.ply` and
`routing.json`) are opened the same way. For each variable the Viewer merges
the experts it is routed to, reading that variable's intensity channel; pick
the variable in the *Variable* combo of the Rendering panel, or start with one:

```bash
SIBR_gaussianViewer_app_rwdi.exe -m <output/multivar/<dataset>/<run>> --variable p
```

Transfer Function Controls

- **Left-click** on the transfer function to add a control point at the cursor position.
- **Right-click** a control point to delete it.
- **Middle-click** a control point to change its color.



### Optional: Build from Source

If you want to build the Viewer yourself, configure and build it from the
`SIBR_viewer` directory:

```bash
cd SIBR_viewer
cmake -S . -B build
cmake --build build --target install --config RelWithDebInfo
```