#!/bin/bash
# Writes logs/jobs237.txt: constrained channel (soft variable) routing.
#   base    channel routing without constraints (where not run before)
#   topk    --topk 1/2 (exact) or --topk 2 (max, i.e. 1..2 experts per variable)
#   budget  --target_mb per dataset (equal split over experts), for transfer /
#           separate / shared inits; also top-2 (max) + budget on separate init
# Outputs: output/multivar/<dataset>/routing2/<name>
cd /home/Paris/smin/ESVR_geo_MoE
O=output/multivar
CH="--routing channel --off_init 0.02"
NEW="--max_primitives 500000 --split_budget --init_num 100000"
SNU="--subvol_size 32 --iterations 1000 --position_lr_max_steps 800 --scaling_lr_max_steps 800 --rotation_lr_max_steps 800 --densification_interval 16 --init_num 30000 --max_primitives 250000 --split_budget --eval_scans 100 400 800 --timestep 0 --lasso_scans 400 800"

# dataset | base args | transfer groups (tag:path) | lambdas | best lambda | budget targets (MB) | extra budget args
DATASETS=(
"ionization_ds2|$NEW|K2:$O/ionization_ds2/separate/affinity/groups_transfer_K2.json|1e-1 3e-1|3e-1|2 4 8|"
"combustion_ds2|$NEW|K2:$O/combustion_ds2/separate/affinity/groups_transfer_K2.json|1e-1 3e-1|3e-1|2 4 8|"
"snu_down_down|$SNU|K4:$O/snu_down_down/C/separate/affinity/groups_transfer_K4.json|1e-2 1e-1|1e-2|4 8 16|--budget_scan 500"
"rocom_masked||K4:$O/rocom_masked/separate/affinity/groups_transfer_K4.json|1e-2 3e-2|1e-2|4 8 16|"
)

job() {  # tag dataset groups args...
    local tag=$1 ds=$2 groups=$3; shift 3
    echo "r2_${ds%%_*}_${tag}|-|./train_multivar.sh $ds $groups $* --output_dir multivar/$ds/routing2/$tag"
}

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r DS BASE TRANSFER LAMBDAS BEST TARGETS BUDGET_EXTRA <<< "$entry"
    TTAG=${TRANSFER%%:*}; TGROUPS=${TRANSFER#*:}
    V=$(python3 -c "from dataset_configs import DATASET_CONFIGS as C; print(len(C['$DS']['variables']))")
    KSEP=K$V
    for L in $LAMBDAS; do
        # base: separate init (transfer init already exists for ROCOM / ionization / combustion)
        job "${KSEP}sep_lam$L" $DS separate $BASE $CH --lasso_lambda $L
        [ "$DS" = snu_down_down ] && job "${TTAG}_lam$L" $DS $TGROUPS $BASE $CH --lasso_lambda $L
        # top-k
        for INIT in "$TTAG:$TGROUPS" "${KSEP}sep:separate"; do
            ITAG=${INIT%%:*}; IGROUPS=${INIT#*:}
            K=${ITAG#K}; K=${K%sep}
            job "${ITAG}_lam${L}_top1" $DS $IGROUPS $BASE $CH --lasso_lambda $L --topk 1 --topk_mode exact
            if [ "$K" -gt 2 ]; then
                job "${ITAG}_lam${L}_top2exact" $DS $IGROUPS $BASE $CH --lasso_lambda $L --topk 2 --topk_mode exact
                job "${ITAG}_lam${L}_top2max" $DS $IGROUPS $BASE $CH --lasso_lambda $L --topk 2 --topk_mode max
            fi
        done
    done
    # budget at the best lambda
    for T in $TARGETS; do
        job "shared_lam${BEST}_mb$T" $DS shared $BASE $CH --lasso_lambda $BEST --target_mb $T $BUDGET_EXTRA
        job "${TTAG}_lam${BEST}_mb$T" $DS $TGROUPS $BASE $CH --lasso_lambda $BEST --target_mb $T $BUDGET_EXTRA
        job "${KSEP}sep_lam${BEST}_mb$T" $DS separate $BASE $CH --lasso_lambda $BEST --target_mb $T $BUDGET_EXTRA
        job "${KSEP}sep_lam${BEST}_top2max_mb$T" $DS separate $BASE $CH --lasso_lambda $BEST --topk 2 --topk_mode max --target_mb $T $BUDGET_EXTRA
    done
done > logs/jobs237.txt
wc -l logs/jobs237.txt
