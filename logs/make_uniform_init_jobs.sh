#!/bin/bash
# Writes logs/jobs246_uniform.txt: channel routing from a uniform init (no prior groups,
# every channel on, different init points per expert) with the same K and lambda as the
# best transfer-K and separate-K=V runs, no budget.
# Outputs: output/multivar/<dataset>/uniform_init/K<K>_lam<lambda>
cd /home/Paris/smin/ESVR_geo_MoE
CH="--routing channel --off_init 0.02"
NEW="--max_primitives 500000 --split_budget --init_num 100000"

# dataset | base args | K:lambda ...  (transfer K first, then K = V)
DATASETS=(
"ionization_ds2|$NEW|2:3e-1 4:1e-1"
"combustion_ds2|$NEW|2:3e-1 4:3e-1"
"rocom_masked||4:1e-2 8:1e-2"
)
for entry in "${DATASETS[@]}"; do
    IFS='|' read -r DS BASE SETTINGS <<< "$entry"
    for S in $SETTINGS; do
        K=${S%%:*}; L=${S#*:}
        echo "ui_${DS%%_*}_K${K}_lam$L|-|./train_multivar.sh $DS shared $BASE $CH --uniform_init $K --lasso_lambda $L --output_dir multivar/$DS/uniform_init/K${K}_lam$L"
    done
done > logs/jobs246_uniform.txt
wc -l logs/jobs246_uniform.txt
