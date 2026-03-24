
#!/bin/bash
MODEL_DIR="saved_models"
DATASET_DIR="./data"
RESULT_DIR="./results"
LOG_DIR="./logs"
BATCH_SIZE=32
DEVICE="cuda"  # or "cpu"


TARGET_METRIC=""  # "mae", "mae_rt", or "ll"，or "" for all

mkdir -p "$RESULT_DIR"
mkdir -p "$LOG_DIR"

echo "Scanning models in $MODEL_DIR..."
if [ -n "$TARGET_METRIC" ]; then
    echo "Only testing models with metric: $TARGET_METRIC"
fi
echo ""

for model_path in "$MODEL_DIR"/*.pth; do
    model_file=$(basename "$model_path")
    
    if [[ ! "$model_file" =~ ^fold[0-9]_ ]]; then
        echo "[SKIP] Not a fold model: $model_file"
        continue
    fi
    
    if [[ "$model_file" =~ fold([0-9])_variation0_(.+)_best_model_best_(.+)\.pth ]]; then
        fold="${BASH_REMATCH[1]}"
        dataset="${BASH_REMATCH[2]}"
        metric="${BASH_REMATCH[3]}"
        if [ -n "$TARGET_METRIC" ] && [ "$metric" != "$TARGET_METRIC" ]; then
            echo "[SKIP] Metric mismatch: $metric != $TARGET_METRIC"
            continue
        fi
        
        fold_dataset="$DATASET_DIR/test_fold${fold}_variation0_${dataset}.csv"
        full_dataset="$DATASET_DIR/${dataset}.csv"
        
        if [ ! -f "$fold_dataset" ]; then
            echo "[SKIP] Test dataset not found: $fold_dataset"
            continue
        fi
        if [ ! -f "$full_dataset" ]; then
            echo "[SKIP] Full dataset not found: $full_dataset"
            continue
        fi
        
        log_file="$LOG_DIR/test_${dataset}_fold${fold}_${metric}.log"
        
        echo "========================================"
        echo "Testing: $model_file"
        echo "  Dataset:      $dataset"
        echo "  Fold:         $fold"
        echo "  Metric:       $metric"
        echo "  Fold dataset: $fold_dataset"
        echo "  Full dataset: $full_dataset"
        echo "========================================"
        
        python test_only.py \
            --fold_dataset "$DATASET_DIR/fold${fold}_variation0_${dataset}.csv" \
            --full_dataset "$full_dataset" \
            --model_path "$model_path" \
            --batch_size "$BATCH_SIZE" \
            --device "$DEVICE" 
            # 2>&1 | tee "$log_file"
        
        if [ $? -eq 0 ]; then
            echo "[OK] Test completed"
        else
            echo "[ERROR] Test failed!"
        fi
        echo ""
    else
        echo "[SKIP] Cannot parse filename: $model_file"
    fi 
done

echo ""
echo "All tests completed!"
echo "Results saved in: $RESULT_DIR"
echo "Logs saved in:    $LOG_DIR"
