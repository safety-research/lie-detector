#!/bin/bash

# Create log directory
mkdir -p lr_exp

# Learning rates to experiment with
LEARNING_RATES=(
    1e-5
    5e-5
    1e-4
    2e-4
    5e-4
    1e-3
    2e-3
)

# Run training for each learning rate sequentially
for lr in "${LEARNING_RATES[@]}"; do
    echo "Starting training with learning rate: $lr"
    
    # Wait for any existing python processes to finish
    while pgrep -f "sft_part3_kfold_trainone" > /dev/null; do
        echo "Waiting for previous experiment to finish..."
        sleep 30
    done
    
    # Run the training script
    bash -c "
        cd ..
        source ~/lies310/bin/activate
        python -u train/sft_part3_kfold_trainone.py --learning_rate $lr
    " > lr_exp/lr_${lr}.log 2>&1
    
    echo "Completed training with learning rate: $lr"
    sleep 10
done

echo "All experiments completed!" 