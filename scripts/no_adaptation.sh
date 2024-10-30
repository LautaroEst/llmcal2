#!/bin/bash -ex

source ./scripts/env.sh

export CUDA_VISIBLE_DEVICES=0,1

for dataset in "${DATASETS[@]}"; do
    # predict on base version
    prediction_list="test_${dataset2testsize[$dataset]}"
    data_path=outputs/prompts/$model/$dataset/all.jsonl
    output_dir="outputs/adaptation/$model/no_adaptation/all/test=$dataset/list=$prediction_list"
    mkdir -p $output_dir
    if [ ! -f $output_dir/logits.csv ]; then
        python -m llmcal2.scripts.run_posteriors \
            --checkpoint_dir $CHECKPOINTS_DIR/${model2checkpoint[$model]} \
            --data_path $data_path \
            --output_dir $output_dir \
            --prediction_lists lists/$dataset/$prediction_list.txt \
            --precision "bf16-true" \
            --devices 2 \
            --num_nodes 1 \
            --batch_size 1 \
            --max_seq_length $max_seq_length
    fi
    # predict on instruct version if exists
    if [ ! -z ${model2checkpoint[${model}-instruct]} ]; then
        output_dir="outputs/adaptation/$model/instruct/all/test=$dataset/list=$prediction_list"
        mkdir -p $output_dir
        if [ ! -f $output_dir/logits.csv ]; then
            python -m llmcal2.scripts.run_posteriors \
                --checkpoint_dir $CHECKPOINTS_DIR/${model2checkpoint[${model}-instruct]} \
                --data_path $data_path \
                --output_dir $output_dir \
                --prediction_lists lists/$dataset/$prediction_list.txt \
                --precision "bf16-true" \
                --devices 2 \
                --num_nodes 1 \
                --batch_size 1 \
                --max_seq_length $max_seq_length
        fi

done