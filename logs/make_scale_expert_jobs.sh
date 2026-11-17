#!/bin/bash
# Writes logs/jobs246_scale.txt: scale-specialized experts (K=3: coarse / mid / fine),
# channel routing from a uniform init, total primitives = the best run of each dataset.
#   A  K=3, same parameters, equal size per expert                  (control)
#   B  A + scale bounds: coarse every axis >= q75, fine <= q25      (scale)
#   C  B + sizes 10:30:60 (coarse:mid:fine; coarse densifies less) (memory)
#   D  C + progressive: coarse from scan 0, mid 100, fine 200      (training order)
# q25 / q75 = quantiles of the max-axis scale (voxels) of the best run's primitives.
# Seed 0 for all runs first, then seed 1.
# Outputs: output/multivar/<dataset>/scale_expert/<exp>_seed<s>
# ionization / combustion (lambda 3e-1): the lasso and dark pruning end the model far
# below its target, so their totals are 0.66 x 500k (grown to 500k as in the original
# runs, not to the best run's final N); outputs scale_expert_g500/.
cd /home/Paris/smin/ESVR_geo_MoE
CH="--routing channel --off_init 0.02 --uniform_init 3"
NEW="--max_primitives 500000 --split_budget --init_num 100000"
SNU="--subvol_size 32 --iterations 1000 --position_lr_max_steps 800 --scaling_lr_max_steps 800 --rotation_lr_max_steps 800 --densification_interval 16 --init_num 30000 --split_budget --eval_scans 100 400 800 --timestep 0 --lasso_scans 400 800 --budget_scan 500"

# dataset | base args | lambda | total primitives of the best run | coarse min | fine max (voxels)
#   ionization  routing2/K4sep_lam1e-1      q25 3.49  q75 5.85
#   combustion  routing2/K4sep_lam3e-1      q25 6.22  q75 11.16
#   ROCOM       channel/K4_off0.02_lam1e-2  q25 2.96  q75 5.65
#   SNU         routing2/K4_lam1e-2         q25 3.51  q75 4.61
DATASETS=(
"ionization_ds2|$NEW|3e-1|330000|5.85|3.49"
"combustion_ds2|$NEW|3e-1|330000|11.16|6.22"
"rocom_masked||1e-2|187665|5.65|2.96"
"snu_down_down|$SNU|1e-2|246074|4.61|3.51"
)

for SEED in 0 1; do
    for entry in "${DATASETS[@]}"; do
        IFS='|' read -r DS BASE L N CMIN FMAX <<< "$entry"
        OUT=scale_expert; [ $N = 330000 ] && OUT=scale_expert_g500
        [ -n "$ONLY_G500" ] && [ $OUT = scale_expert ] && continue
        EQUAL=$(python3 -c "print(*[round($N / 3)] * 3)")
        RATIO=$(python3 -c "print(*[round($N * r) for r in (0.1, 0.3, 0.6)])")
        BOUNDS="--scale_bounds $CMIN:0 0:0 0:$FMAX"
        for EXP in A B C D; do
            case $EXP in
                A) X="--expert_target $EQUAL" ;;
                B) X="--expert_target $EQUAL $BOUNDS" ;;
                C) X="--expert_target $RATIO $BOUNDS" ;;
                D) X="--expert_target $RATIO $BOUNDS --progressive_scans 0 100 200" ;;
            esac
            echo "se_${DS%%_*}_${OUT#scale_expert}${EXP}_s$SEED|-|./train_multivar.sh $DS shared $BASE $CH --lasso_lambda $L $X --seed $SEED --output_dir multivar/$DS/$OUT/${EXP}_seed$SEED"
        done
    done
done > logs/jobs246_scale.txt
wc -l logs/jobs246_scale.txt
