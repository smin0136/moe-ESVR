#!/bin/bash
# Writes logs/jobs201_topkcap.txt: top-k routing with more capacity per expert.
# Starting from each dataset's best unconstrained channel-routing run, every
# expert may keep s x (mean primitives per expert of that run) primitives
# (--expert_cap, applied at the global prune / --budget_scan), s = 1.5 and 2.
# Per s: no limit (control: same capacity, no top-k) / top1 / top2 exact / top2 max.
# Outputs: output/multivar/<dataset>/topk_cap/s<s>_<variant>
cd /home/Paris/smin/ESVR_geo_MoE
O=output/multivar
CH="--routing channel --off_init 0.02"
NEW="--max_primitives 500000 --split_budget --init_num 100000"
SNU="--subvol_size 32 --iterations 1000 --position_lr_max_steps 800 --scaling_lr_max_steps 800 --rotation_lr_max_steps 800 --densification_interval 16 --init_num 30000 --split_budget --eval_scans 100 400 800 --timestep 0 --lasso_scans 400 800 --budget_scan 500"

# dataset | groups | base args | lambda | mean primitives per expert of the best unconstrained run
#   ionization  routing2/K4sep_lam1e-1          [40405, 29709, 38953, 32570]
#   combustion  routing2/K4sep_lam3e-1          [18594, 21780, 26414, 25096]
#   ROCOM       channel/K4_off0.02_lam1e-2      [10070, 39483, 117338, 20774]
#   SNU         routing2/K4_lam1e-2             [31082, 30692, 31000, 153300]
DATASETS=(
"ionization_ds2|separate|$NEW|1e-1|35409"
"combustion_ds2|separate|$NEW|3e-1|22971"
"rocom_masked|$O/rocom_masked/separate/affinity/groups_transfer_K4.json||1e-2|46916"
"snu_down_down|$O/snu_down_down/C/separate/affinity/groups_transfer_K4.json|$SNU|1e-2|61519"
)

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r DS GRP BASE L MEAN <<< "$entry"
    for S in 1.5 2; do
        CAP=$(python3 -c "print(round($S * $MEAN))")
        EXTRA="--expert_cap $CAP"
        # SNU: split densify cap is 1/8 of the total for single-variable experts, so
        # raise the total until every expert can densify up to the new cap.
        [ "$DS" = snu_down_down ] && EXTRA="$EXTRA --max_primitives $((8 * CAP))"
        for V in nolimit top1 top2exact top2max; do
            case $V in
                nolimit)   TOPK="" ;;
                top1)      TOPK="--topk 1 --topk_mode exact" ;;
                top2exact) TOPK="--topk 2 --topk_mode exact" ;;
                top2max)   TOPK="--topk 2 --topk_mode max" ;;
            esac
            TAG="s${S}_$V"
            echo "tc_${DS%%_*}_$TAG|-|./train_multivar.sh $DS $GRP $BASE $CH --lasso_lambda $L $EXTRA $TOPK --output_dir multivar/$DS/topk_cap/$TAG"
        done
    done
done > logs/jobs201_topkcap.txt
wc -l logs/jobs201_topkcap.txt
