python dataset.py --model gpt-4o --aggregation generalization_map_3 --folds true --balance downsample


python -m prep.train --dataset .data/openai/gpt_4o/mask --model gpt-4o-2024-08-06 --size 128

python train.py --dataset .data/openai/gpt_4o/in_context_pressure --model gpt-4o-2024-11-20 --size 16


python -m safetytooling.apis.finetuning.openai.run \
    --model 'gpt-3.5-turbo-1106' \
    --train_file data/train.jsonl \
    --val_file data/val.jsonl \
    --n_epochs 3 \
    --learning_rate_multiplier 0.5 \
    --batch_size 8