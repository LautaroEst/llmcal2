#!/bin/bash -ex

source ./scripts/env.sh

precision="bf16-true"

declare -A model_dirs=(
    ["no_adaptation"]=$CHECKPOINTS_DIR/${model2checkpoint[$model]}
    ["instruct"]=$CHECKPOINTS_DIR/${model2checkpoint[${model}-instruct]}
)

for dataset in "${DATASETS[@]}"; do
    for model_type in no_adaptation instruct; do
        model_dir=${model_dirs[$model_type]}
        if [ -z $model_dir ]; then
            continue
        fi
    
        # Predictions directories and lists
        data_path=outputs/prompts/$model/$dataset/all.jsonl
        test_list="test_${dataset2testsize[$dataset]}"
        base_output_dir="outputs/adaptation/$model/$model_type/all"
        test_dir="$base_output_dir/test=$dataset/list=$test_list"
        if [ ! -f $test_dir/logits.csv ]; then
            mkdir -p $test_dir
            python -m llmcal2.scripts.run_posteriors \
                --checkpoint_dir $model_dir \
                --data_path $data_path \
                --output_dir $test_dir \
                --prediction_lists lists/$dataset/$test_list.txt \
                --precision $precision \
                --devices 1 \
                --num_nodes 1 \
                --batch_size 1 \
                --max_seq_length $max_seq_length
        fi
    done
done