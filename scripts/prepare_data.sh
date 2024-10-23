#!/bin/bash -ex

source ./scripts/env.sh

for dataset in ${DATASETS[@]}; do
    output_path=outputs/prompts/$model/$dataset/all.jsonl
    if [ ! -f $output_path ] ; then
        mkdir -p $(dirname $output_path)
        python -m llmcal2.scripts.prepare_data \
            --dataset_path data/$dataset/all.csv \
            --prompt_template prompts/basic_$dataset.yaml  \
            --model $model \
            --output_path $output_path  \
            --max_characters 600
    fi
done