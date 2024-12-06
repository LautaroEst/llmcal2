#!/bin/bash -ex

source ./scripts/env.sh
export CUDA_VISIBLE_DEVICES=0

precision="bf16-true"

declare -A model_dirs=(
    ["no_adaptation"]=$CHECKPOINTS_DIR/${model2checkpoint[$model]}
    ["instruct"]=$CHECKPOINTS_DIR/${model2checkpoint[${model}-instruct]}
)
val_prop=0.3

tolerance=1e-5

for dataset in "${DATASETS[@]}"; do
    for num_seed in $(seq 0 $((num_seeds-1))); do
        for model_type in no_adaptation instruct; do
            model_dir=${model_dirs[$model_type]}
            if [ -z $model_dir ]; then
                continue
            fi
            
            seed=$((base_seed+num_seed))
            data_path=outputs/prompts/$model/$dataset/all.jsonl
            test_list="test_${dataset2testsize[$dataset]}"
            val_list="val_${dataset2trainsize[$dataset]}_${val_prop}_$num_seed"
            train_list="train_${dataset2trainsize[$dataset]}_0.0_$num_seed"

            # Predictions directories and lists
            base_output_dir="outputs/adaptation/$model/$model_type/all"        
            prediction_dirs=(
                "$base_output_dir/test=$dataset/list=$val_list"
                "$base_output_dir/test=$dataset/list=$train_list"
            )
            prediction_lists=(
                "$val_list"
                "$train_list"
            )
            for i in $(seq 0 1); do
                prediction_dir=${prediction_dirs[$i]}
                prediction_list=${prediction_lists[$i]}
                if [ ! -f $prediction_dir/logits.csv ]; then
                    mkdir -p $prediction_dir
                    python -m llmcal2.scripts.run_posteriors \
                        --checkpoint_dir $model_dir \
                        --data_path $data_path \
                        --output_dir $prediction_dir \
                        --prediction_lists lists/$dataset/$prediction_list.txt \
                        --precision $precision \
                        --devices 1 \
                        --num_nodes 1 \
                        --batch_size 1 \
                        --max_seq_length $max_seq_length
                fi
            done

            # Calibration directories
            cal_dir="outputs/adaptation/$model/${model_type}_plus_iterativecal/${dataset}_${dataset2trainsize[$dataset]}_0.0_$num_seed"
            if [ ! -f "$cal_dir/test=$dataset/list=$test_list/logits.csv" ]; then
                mkdir -p $cal_dir/test=$dataset/list=$test_list
                python -m llmcal2.scripts.iterative_calibration \
                    --output_dir $cal_dir/test=$dataset/list=$test_list \
                    --train_alpha_logits $base_output_dir/test=$dataset/list=$val_list/logits.csv \
                    --train_alpha_labels $base_output_dir/test=$dataset/list=$val_list/labels.csv \
                    --train_beta_logits $base_output_dir/test=$dataset/list=$train_list/logits.csv \
                    --train_beta_labels $base_output_dir/test=$dataset/list=$train_list/labels.csv \
                    --predict_logits $base_output_dir/test=$dataset/list=$test_list/logits.csv \
                    --predict_labels $base_output_dir/test=$dataset/list=$test_list/labels.csv \
                    --tolerance $tolerance
            fi
        done
    done
done