#!/bin/bash
# Writes logs/jobs246_budget.txt: equal-memory comparison with an exact budget.
# --budget_mode exact prunes all experts together to --target_mb at the end of the
# penalty window (scan 2000), counting only the channels that stay, with an equal
# share per expert (unused shares redistributed). It replaces the fixed-ratio
# global prune for every method, so all runs end at the same size.
#   shared   one expert + channel lasso
#   separate hard one-to-one experts, no lasso
#   routing  the two channel-routing inits used before (transfer K / separate K=V)
# Outputs: output/multivar/<dataset>/budget_exact/<method>_mb<T>
cd /home/Paris/smin/ESVR_geo_MoE
O=output/multivar
CH="--routing channel --off_init 0.02"
NEW="--max_primitives 500000 --split_budget --init_num 100000"

# dataset | base args | shared/routing lambda | transfer groups (tag:path) | separate-K lambda | targets (MB)
DATASETS=(
"ionization_ds2|$NEW|3e-1|K2:$O/ionization_ds2/separate/affinity/groups_transfer_K2.json|1e-1|2 4 8"
"combustion_ds2|$NEW|3e-1|K2:$O/combustion_ds2/separate/affinity/groups_transfer_K2.json|3e-1|2 4 8"
"rocom_masked||1e-2|K4:$O/rocom_masked/separate/affinity/groups_transfer_K4.json|1e-2|4 8 16"
)

job() {  # tag dataset groups args...
    local tag=$1 ds=$2 groups=$3; shift 3
    echo "be_${ds%%_*}_${tag}|-|./train_multivar.sh $ds $groups $* --output_dir multivar/$ds/budget_exact/$tag"
}

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r DS BASE L TRANSFER LSEP TARGETS <<< "$entry"
    TTAG=${TRANSFER%%:*}; TGROUPS=${TRANSFER#*:}
    V=$(python3 -c "from dataset_configs import DATASET_CONFIGS as C; print(len(C['$DS']['variables']))")
    for T in $TARGETS; do
        B="--budget_mode exact --target_mb $T"
        job "shared_mb$T"            $DS shared     $BASE $CH --lasso_lambda $L $B
        job "separate_mb$T"          $DS separate   $BASE $B
        job "${TTAG}_lam${L}_mb$T"   $DS $TGROUPS   $BASE $CH --lasso_lambda $L $B
        job "K${V}sep_lam${LSEP}_mb$T" $DS separate $BASE $CH --lasso_lambda $LSEP $B
    done
done > logs/jobs246_budget.txt
wc -l logs/jobs246_budget.txt
