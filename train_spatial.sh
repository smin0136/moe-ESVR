#!/bin/bash

# Train multivariate ESVR with spatial x variable routing (attribute MoE).
# Usage: ./train_spatial.sh <dataset_name> <groups> [extra train_spatial.py args]
#   groups: shared | path/to/groups.json (G shape experts)
DATASET_NAME="${1:-rocom_masked}"
VAR_GROUPS="${2:-shared}"
shift $(( $# < 2 ? $# : 2 ))
EXTRA_ARGS=("$@")

SUBVOL_SIZE=128

# ============================================================
# Hyperparameters (scan-based, same as train_chunk.sh)
# ============================================================
ITERATIONS_SCAN=2800
DENSIFY_FROM_SCAN=16
DENSIFY_UNTIL_SCAN=400
DENSIFICATION_INTERVAL_SCAN=4
OPACITY_RESET_INTERVAL_SCAN=64
DARKPRUNING_SCANS="16 200 400 1200 1800 2100 2700"
GLOBALPRUNING_SCANS="1200"
EVAL_SCANS="400 1200 2000"

POSITION_LR_MAX_SCANS=1200
OPACITY_LR_MAX_SCANS=16
SCALING_LR_MAX_SCANS=1200
ROTATION_LR_MAX_SCANS=1200

POSITION_LR_INIT=0.000008
POSITION_LR_FINAL=0.000001
OPACITY_LR_INIT=0.0075
OPACITY_LR_FINAL=0.0005
SCALING_LR_INIT=0.00005
SCALING_LR_FINAL=0.00005
ROTATION_LR_INIT=0.00001
ROTATION_LR_FINAL=0.00001
POSITION_LR_DELAY_MULT=0.01
LAMBDA_DSSIM=0.2

DENSIFY_GRAD_THRESHOLD=0.0000000025
PERCENT_DENSE=0.01

INIT_NUM=100000
PRUNING_REMAINWEIGHT=0.66

GROUPS_TAG=$(basename "${VAR_GROUPS}" .json)
OUTPUT_DIR="spatial/${DATASET_NAME}/${GROUPS_TAG}"

python train_spatial.py \
    --dataset_name ${DATASET_NAME} \
    --groups ${VAR_GROUPS} \
    --output_dir ${OUTPUT_DIR} \
    --subvol_size ${SUBVOL_SIZE} \
    --iterations ${ITERATIONS_SCAN} \
    --densify_from_iter ${DENSIFY_FROM_SCAN} \
    --densify_until_iter ${DENSIFY_UNTIL_SCAN} \
    --densification_interval ${DENSIFICATION_INTERVAL_SCAN} \
    --opacity_reset_interval ${OPACITY_RESET_INTERVAL_SCAN} \
    --darkpruning_iter ${DARKPRUNING_SCANS} \
    --globalpruning_iter ${GLOBALPRUNING_SCANS} \
    --eval_scans ${EVAL_SCANS} \
    --position_lr_max_steps ${POSITION_LR_MAX_SCANS} \
    --opacity_lr_max_steps ${OPACITY_LR_MAX_SCANS} \
    --scaling_lr_max_steps ${SCALING_LR_MAX_SCANS} \
    --rotation_lr_max_steps ${ROTATION_LR_MAX_SCANS} \
    --position_lr_init ${POSITION_LR_INIT} \
    --position_lr_final ${POSITION_LR_FINAL} \
    --opacity_lr_init ${OPACITY_LR_INIT} \
    --opacity_lr_final ${OPACITY_LR_FINAL} \
    --scaling_lr_init ${SCALING_LR_INIT} \
    --scaling_lr_final ${SCALING_LR_FINAL} \
    --rotation_lr_init ${ROTATION_LR_INIT} \
    --rotation_lr_final ${ROTATION_LR_FINAL} \
    --position_lr_delay_mult ${POSITION_LR_DELAY_MULT} \
    --lambda_dssim ${LAMBDA_DSSIM} \
    --densify_grad_threshold ${DENSIFY_GRAD_THRESHOLD} \
    --percent_dense ${PERCENT_DENSE} \
    --init_num ${INIT_NUM} \
    --pruning_remainweight ${PRUNING_REMAINWEIGHT} \
    "${EXTRA_ARGS[@]}"
