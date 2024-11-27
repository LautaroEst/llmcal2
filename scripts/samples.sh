#!/bin/bash -ex

source ./scripts/env.sh

precision="bf16-true"

declare -A dataset2trainsizes=(
    ["sst2"]="256 512 1024"
    ["agnews"]="256 512 1024"
    ["dbpedia"]="448 896 1792"
    ["20newsgroups"]="640 1280 2560"
    ["banking77"]="1232 2464 4928"
)

val_prop=0.3

model_dir=$CHECKPOINTS_DIR/${model2checkpoint[$model]}

for dataset in "${!dataset2trainsizes[@]}"; do
    for method in lora_multiple no_adaptation_plus_dp_cal; do
    for num_seed in $(seq 0 $((num_seeds-1))); do
        for size in ${dataset2trainsizes[$dataset]}; do
            seed=$((base_seed+num_seed))
            data_path=outputs/prompts/$model/$dataset/all.jsonl
            test_list="test_${dataset2trainsizes[$dataset]}"
            train_list="val_${size}_0.3_${num_seed}"
            train_output_dir=outputs/adaptation/$model/$method/${dataset}_${size}_0.3_${num_seed}
            output_dir="$train_output_dir/test=$dataset/list=$test_list"
            mkdir -p $output_dir
            if [ ! -f $output_dir/logits.csv ]; then
                
                ## TRAIN LORA
                if [[ $method == "lora_multiple" ]]; then
                    output_checkpoint_dir=$train_output_dir/checkpoint_final
                    python -m llmcal2.scripts.train_lora_multiple
                    
                    if [ ! -f $output_dir/logits.csv ]; then
                        mkdir -p $output_dir
                        python -m llmcal2.scripts.run_posteriors \
                            --checkpoint_dir $output_checkpoint_dir \
                            --data_path $data_path \
                            --output_dir $output_dir \
                            --prediction_lists lists/$dataset/$test_list.txt \
                            --precision $precision \
                            --devices 1 \
                            --num_nodes 1 \
                            --batch_size 1 \
                            --max_seq_length $max_seq_length
                    fi
                
                ## TRAIN DPCAL
                elif [[ $method == "no_adaptation_plus_dp_cal" ]]; then
                    prediction_dir="outputs/adaptation/$model/no_adaptation/all/test=$dataset/list=$val_list"
                    if [ ! -f $prediction_dir/logits.csv ]; then
                        mkdir -p $prediction_dir
                        python -m llmcal2.scripts.run_posteriors \
                            --checkpoint_dir $model_dir \
                            --data_path $data_path \
                            --output_dir $prediction_dir \
                            --prediction_lists lists/$dataset/$train_list.txt \
                            --precision $precision \
                            --devices 1 \
                            --num_nodes 1 \
                            --batch_size 1 \
                            --max_seq_length $max_seq_length
                    fi
                    mkdir -p $output_dir $train_output_dir/logs
                    python -m llmcal2.scripts.affine_calibration \
                        --output_dir $output_dir \
                        --log_dir $train_output_dir/logs \
                        --checkpoint_dir $train_output_dir \
                        --train_logits $prediction_dir/logits.csv \
                        --train_labels $prediction_dir/labels.csv \
                        --predict_logits $output_dir/logits.csv \
                        --predict_labels $output_dir/labels.csv \
                        --method "dp_calibration" \
                        --learning_rate 1e-3 \
                        --tolerance 1e-5 \
                        --max_ls 40 \
                        --seed $seed
                else
                    echo "Unknown method: $method"
                    exit 1
                fi
            fi
        done
    done
done
            