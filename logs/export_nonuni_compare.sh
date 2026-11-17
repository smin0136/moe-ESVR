#!/bin/bash
# TIF exports of the uniform (exact) vs non-uniform budget runs at the middle target
# into output/nonuni_compare/<dataset>/<method>_<mode>/. Usage: logs/export_nonuni_compare.sh <dataset>
cd /home/Paris/smin/ESVR_geo_MoE
O=output/multivar
DS=$1
case $DS in
    ionization_ds2) T=4; METHODS="separate K2_lam3e-1 K4sep_lam1e-1" ;;
    combustion_ds2) T=4; METHODS="separate K2_lam3e-1 K4sep_lam3e-1" ;;
    rocom_masked)   T=8; METHODS="separate K4_lam1e-2 K8sep_lam1e-2" ;;
esac
RUNS=()
for M in $METHODS; do
    RUNS+=(--run ${M}_exact=$O/$DS/budget_exact/${M}_mb$T)
    for MODE in pool usage ratio; do
        RUNS+=(--run ${M}_$MODE=$O/$DS/budget_nonuni/${M}_mb${T}_$MODE)
    done
done
python export_multivar_tif.py --out output/nonuni_compare/$DS "${RUNS[@]}"
