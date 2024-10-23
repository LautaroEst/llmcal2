#!/bin/bash -ex

source ./scripts/env.sh

for dataset in "${DATASETS[@]}"; do
    prediction_list="test_${dataset2testsize[$dataset]}"
    data_path=outputs/prompts/$model/$dataset/all.jsonl
    output_dir=outputs/adaptation/$model/no_adaptation/$dataset/$prediction_list
    mkdir -p $output_dir
    if [ ! -f $output_dir/logits.csv ]; then
        python -m llmcal2.scripts.run_posteriors \
            --checkpoint_dir $CHECKPOINTS_DIR/${model2checkpoint[$model]} \
            --data_path $data_path \
            --output_dir $output_dir \
            --prediction_list lists/$dataset/$prediction_list.txt \
            --precision "bf16-true" \
            --devices 1 \
            --num_nodes 1 \
            --batch_size 1 \
            --max_seq_length $max_seq_length
    fi
done