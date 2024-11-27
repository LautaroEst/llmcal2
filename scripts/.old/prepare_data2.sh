#!/bin/bash -ex

source ./scripts/env2.sh

max_characters=600

for dataset in ${DATASETS[@]}; do
    output_path=outputs/prompts/$model/$dataset/all.jsonl
    if [ ! -f $output_path ] ; then
        mkdir -p $(dirname $output_path)
        python -m llmcal2.scripts.prepare_data \
            --dataset_path data/$dataset/all.csv \
            --prompt_template prompts/basic_$dataset.yaml  \
            --model $model \
            --output_path $output_path  \
            --max_characters $max_characters
    fi

    for n_shots in ${N_SHOTS[@]}; do
        for num_seed in $(seq 0 $((num_seeds-1))); do
            shots_list=${n_shots}shots_${num_seed}
            output_path=outputs/prompts/$model/$dataset/$shots_list.jsonl
            if [ ! -f $output_path ] ; then
                mkdir -p $(dirname $output_path)
                python -m llmcal2.scripts.prepare_data \
                    --dataset_path data/$dataset/all.csv \
                    --prompt_template prompts/basic_$dataset.yaml  \
                    --model $model \
                    --output_path $output_path  \
                    --shots_list lists/$dataset/$shots_list.txt \
                    --max_characters $max_characters
            fi
        done
    done
done