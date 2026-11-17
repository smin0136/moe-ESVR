#!/bin/bash
# Writes logs/jobs_topk_target.txt: top-k routing at matched per-variable capacity.
# Each variant retrains with a routing fixed from the start (--fixed_routing; no
# late cut), taken from the runs that learned it: no limit = the best
# unconstrained run, top1 / top2 exact / top2 max = the routing2 top-k runs at the
# same lambda. Every used expert is grown to and pruned at an exact size
# (--expert_target), S = s x mean primitives per expert of the best run:
#   no limit, top2 exact, top2 max: S per expert;  top1: 2S per expert
# (one expert per variable instead of two). s = 1.5, 2.
# Outputs: output/multivar/<dataset>/topk_target/s<s>_<variant>
cd /home/Paris/smin/ESVR_geo_MoE
O=output/multivar
NEW="--max_primitives 500000 --split_budget --init_num 100000"
SNU="--subvol_size 32 --iterations 1000 --position_lr_max_steps 800 --scaling_lr_max_steps 800 --rotation_lr_max_steps 800 --densification_interval 16 --init_num 30000 --split_budget --eval_scans 100 400 800 --timestep 0 --lasso_scans 400 800 --budget_scan 500"

# dataset | base args | mean primitives per expert of the best run | no-limit run | top-k run prefix
DATASETS=(
"rocom_masked||46916|$O/rocom_masked/channel/K4_off0.02_lam1e-2|$O/rocom_masked/routing2/K4_lam1e-2"
"snu_down_down|$SNU|61519|$O/snu_down_down/routing2/K4_lam1e-2|$O/snu_down_down/routing2/K4_lam1e-2"
"ionization_ds2|$NEW|35409|$O/ionization_ds2/routing2/K4sep_lam1e-1|$O/ionization_ds2/routing2/K4sep_lam1e-1"
"combustion_ds2|$NEW|22971|$O/combustion_ds2/routing2/K4sep_lam3e-1|$O/combustion_ds2/routing2/K4sep_lam3e-1"
)

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r DS BASE MEAN NOLIMIT TOPK <<< "$entry"
    K=$(python3 -c "import json; print(len(json.load(open('$NOLIMIT/routing.json'))['groups']))")
    for S in 1.5 2; do
        for V in nolimit top1 top2exact top2max; do
            SRC=$TOPK"_$V"; MULT=1
            [ $V = nolimit ] && SRC=$NOLIMIT
            [ $V = top1 ] && MULT=2
            T=$(python3 -c "print(round($MULT * $S * $MEAN))")
            TARGETS=$(for k in $(seq $K); do printf "%s " $T; done)
            echo "tt_${DS%%_*}_s${S}_$V|-|./train_multivar.sh $DS separate $BASE --routing channel --lasso_lambda 0 --fixed_routing $SRC/metrics.json --expert_target $TARGETS--output_dir multivar/$DS/topk_target/s${S}_$V"
        done
    done
done > logs/jobs_topk_target.txt
wc -l logs/jobs_topk_target.txt
