#!/bin/bash
# Writes logs/jobs246_nonuni.txt: non-uniform memory allocation over experts at an
# exact total budget (same datasets, lambdas and targets as make_budget_exact_jobs.sh,
# whose budget_exact/ runs are the uniform baseline).
#   pool   one budget, primitives compete by importance per stored float
#   usage  shares proportional to each expert's summed active gate (channel RMS)
#   ratio  fixed 60:30:10 shares, largest to the most used expert (others get 10)
# Methods: separate (hard), transfer K, separate K=V with channel routing.
# Middle target first. Outputs: output/multivar/<dataset>/budget_nonuni/<method>_mb<T>_<mode>
cd /home/Paris/smin/ESVR_geo_MoE
O=output/multivar
CH="--routing channel --off_init 0.02"
NEW="--max_primitives 500000 --split_budget --init_num 100000"

# dataset | base args | routing lambda | transfer groups (tag:path) | separate-K lambda | targets (MB, middle first)
DATASETS=(
"rocom_masked||1e-2|K4:$O/rocom_masked/separate/affinity/groups_transfer_K4.json|1e-2|8 4 16"
"ionization_ds2|$NEW|3e-1|K2:$O/ionization_ds2/separate/affinity/groups_transfer_K2.json|1e-1|4 2 8"
"combustion_ds2|$NEW|3e-1|K2:$O/combustion_ds2/separate/affinity/groups_transfer_K2.json|3e-1|4 2 8"
)

job() {  # tag dataset groups args...
    local tag=$1 ds=$2 groups=$3; shift 3
    echo "bn_${ds%%_*}_${tag}|-|./train_multivar.sh $ds $groups $* --output_dir multivar/$ds/budget_nonuni/$tag"
}

for i in 0 1 2; do
    for entry in "${DATASETS[@]}"; do
        IFS='|' read -r DS BASE L TRANSFER LSEP TARGETS <<< "$entry"
        TTAG=${TRANSFER%%:*}; TGROUPS=${TRANSFER#*:}
        V=$(python3 -c "from dataset_configs import DATASET_CONFIGS as C; print(len(C['$DS']['variables']))")
        T=$(cut -d' ' -f$((i + 1)) <<< "$TARGETS")
        for M in pool usage ratio; do
            B="--budget_mode $M --target_mb $T"
            job "separate_mb${T}_$M"            $DS separate $BASE $B
            job "${TTAG}_lam${L}_mb${T}_$M"     $DS $TGROUPS $BASE $CH --lasso_lambda $L $B
            job "K${V}sep_lam${LSEP}_mb${T}_$M" $DS separate $BASE $CH --lasso_lambda $LSEP $B
        done
    done
done > logs/jobs246_nonuni.txt
wc -l logs/jobs246_nonuni.txt
