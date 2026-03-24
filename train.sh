#!/bin/bash

# you can modify this list to run on different datasets, but make sure to have the corresponding fold and full dataset CSV files in the `data/` directory. The fold dataset should be named `fold{i}_variation0_{dataset_name}.csv` and the full dataset should be named `{dataset_name}.csv`.

DATASETS=(
    # "BPI_Challenge_2012"
    "BPI_Challenge_2012_A"
    # "BPI_Challenge_2012_Complete"
    # "BPI_Challenge_2012_O"
    # "BPI_Challenge_2012_W"
    # "BPI_Challenge_2012_W_Complete"
    # "BPI_Challenge_2013_closed_problems"
    # "BPI_Challenge_2013_incidents"
    # "env_permit"
    # "Helpdesk"
    # "nasa"
    # "SEPSIS"
)

MAX_JOBS_PER_GPU=4
i=0

# wait until current GPU has a free slot
wait_for_gpu_slot() {
    local gpu_id=$1
    while true; do
        running=$(screen -ls | grep "thp_" | wc -l)
        gpu_tasks=$(ps aux | grep "CUDA_VISIBLE_DEVICES=${gpu_id}" | grep python | wc -l)
        if [ $gpu_tasks -lt $MAX_JOBS_PER_GPU ]; then
            break
        fi    
        echo "GPU $gpu_id is full ($gpu_tasks/$MAX_JOBS_PER_GPU tasks), waiting..."
        sleep 10
    done
}

for name in "${DATASETS[@]}"; do
    GPU_ID=$((i % 2))  # shift 2 GPUs (0, 1, 0, 1)
    
    wait_for_gpu_slot $GPU_ID
    echo "boot: $name on GPU $GPU_ID"
    
    screen -dmS "thp_${name}" bash -c "
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        exec > logs/${name}_gpu${GPU_ID}.log 2>&1
        echo 'Dataset: $name'
        echo 'Physical GPU: $GPU_ID'
        echo 'CUDA_VISIBLE_DEVICES: '\$CUDA_VISIBLE_DEVICES
        echo '========================='
        python main.py \
            --fold_dataset data/fold0_variation0_${name}.csv \
            --full_dataset data/${name}.csv \
            --train \
            --test \
            --epoch 1 \
            --batch_size 32 \
            --lr 0.01 \
            --device cuda \
            --log_dir runs
    "
    i=$((i + 1))
    
    sleep 2
done



