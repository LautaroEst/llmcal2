#!/bin/bash -ex

source ./scripts/env.sh

lora_args="--lora_r=8 --lora_alpha=16 --lora_dropout=0.05 --lora_query --lora_key --lora_value --lora_projection --lora_mlp --lora_head"
val_prop=0.3
global_batch_size=8
micro_batch_size=1
learning_rate=0.0001
optimizer="adamw"
weight_decay=0.0
patience=10
precision="bf16-true"

for dataset in ${DATASETS[@]}; do
    for num_seed in $(seq 0 $((num_seeds-1))); do
        for n_shot in "${N_SHOTS[@]}"; do
            for loss in ${losses_options[@]}; do
                
                # Seed
                seed=$((base_seed+num_seed))
                
                # Total train samples
                total_train_samples=${dataset2trainsize[$dataset]}

                # Lists
                list_identifier=""
                for d in ${DATASETS[@]}; do
                    if [ $d != $dataset ]; then
                        list_identifier+="${d}_${dataset2trainsize[$d]}_${val_prop}_${num_seed}__"
                    fi
                done
                list_identifier=$(echo "$list_identifier" | sed 's/__$//')
                shots_list=${n_shot}shots_${num_seed}
                data_path="outputs/prompts/$model/$dataset/$shots_list.jsonl"
                cal_list="val_${total_train_samples}_${val_prop}_${num_seed}"
                test_list="test_${dataset2testsize[$dataset]}"
                
                # Train directories
                train_output_dir="outputs/adaptation/$model/lora_${loss}_few_shot/${list_identifier}__${n_shot}shots_${num_seed}"
                log_dir=$train_output_dir/logs
                output_checkpoint_dir=outputs/adaptation/$model/lora_${loss}/$list_identifier/checkpoint

                # Predictions directories
                predictions_dirs=("$train_output_dir/test=$dataset/list=$test_list" "$train_output_dir/test=$dataset/list=$cal_list")
                predictions_lists=("$test_list" "$cal_list")

                # Model directory
                model_dir=$CHECKPOINTS_DIR/${model2checkpoint[$model]}

                # Predict
                for i in ${!predictions_dirs[@]}; do
                    prediction_dir=${predictions_dirs[$i]}
                    prediction_list=${predictions_lists[$i]}
                    if [ ! -f $prediction_dir/logits.csv ]; then
                        mkdir -p $prediction_dir
                        python -m llmcal2.scripts.run_posteriors \
                            --checkpoint_dir $output_checkpoint_dir \
                            --peft "lora" \
                            --data_path $data_path \
                            --output_dir $prediction_dir \
                            --prediction_lists lists/$dataset/$prediction_list.txt \
                            --precision $precision \
                            --devices 1 \
                            --num_nodes 1 \
                            --batch_size 1 \
                            --max_seq_length $inference_max_seq_len \
                            $lora_args
                    fi
                done
            done
        done
    done
done
