#!/bin/bash
# Usage (on any server's host, inside tmux): [QUEUE=jobs237.txt] logs/worker_q3.sh <gpu> [container] [pythonpath]
# Waits until <gpu> uses at most MAX_USED_MIB of memory (other users' small jobs
# may share it), then pops the first runnable
# "tag|dependency|command" line from logs/jobs201.txt and runs the command inside
# <container> (default smin_moe). dependency is a path that must exist first
# ("-" for none). pythonpath is prepended for containers without the CUDA
# extensions in their user site (e.g. smin_gs on 246 uses ext_site/).
GPU=$1
CONTAINER=${2:-smin_moe}
PYPATH=${3:-}
REPO=/home/Paris/smin/ESVR_geo_MoE
Q=$REPO/logs/${QUEUE:-jobs201.txt}
HOST=$(hostname)
cd $REPO
MAX_USED_MIB=${MAX_USED_MIB:-4096}
while true; do
    # A hung nvidia-smi (seen on 246) counts as busy.
    until used=$(timeout 30 nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits) && [ "$used" -le "$MAX_USED_MIB" ]; do
        echo "[$HOST GPU $GPU] busy (${used:-no answer} MiB used), waiting"; sleep 60
    done
    JOB=$(flock $Q.lock bash -c '
        n=0
        while IFS= read -r line; do
            n=$((n+1)); dep=$(cut -d"|" -f2 <<<"$line")
            if [ "$dep" = "-" ] || [ -e "$dep" ]; then echo "$line"; sed -i "${n}d" "$0"; exit; fi
        done < "$0"' $Q)
    if [ -z "$JOB" ]; then
        [ -s $Q ] || { echo "[$HOST GPU $GPU] queue empty"; break; }
        sleep 60; continue
    fi
    TAG=$(cut -d"|" -f1 <<<"$JOB"); CMD=$(cut -d"|" -f3- <<<"$JOB")
    echo "[$HOST GPU $GPU] $(date +%m-%d\ %H:%M) start $TAG" | tee -a logs/workers_${QUEUE:-jobs201.txt}.log
    docker exec -it -w $REPO -e CUDA_VISIBLE_DEVICES=$GPU -e PYTHONUNBUFFERED=1 ${PYPATH:+-e PYTHONPATH=$PYPATH} \
        $CONTAINER bash -c "$CMD" 2>&1 | tee logs/$TAG.log
    echo "[$HOST GPU $GPU] $(date +%m-%d\ %H:%M) done $TAG (exit ${PIPESTATUS[0]})" | tee -a logs/workers_${QUEUE:-jobs201.txt}.log
done
