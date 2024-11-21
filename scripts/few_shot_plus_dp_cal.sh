#!/bin/bash -ex

source ./scripts/env.sh

precision="bf16-true"

declare -A model_dirs=(
    ["no_adaptation"]=$CHECKPOINTS_DIR/${model2checkpoint[$model]}
    ["instruct"]=$CHECKPOINTS_DIR/${model2checkpoint[${model}-instruct]}
)
val_prop=0.3

method="dp_calibration"
learning_rate=1e-3
tolerance=1e-5
max_ls=40

for dataset in "${DATASETS[@]}"; do
    for n_shot in "${N_SHOTS[@]}"; do
        for num_seed in $(seq 0 $((num_seeds-1))); do
            for model_type in no_adaptation instruct; do
                model_dir=${model_dirs[$model_type]}
                if [ -z $model_dir ]; then
                    continue
                fi
                
                seed=$((base_seed+num_seed))
                shots_list=${n_shot}shots_${num_seed}
                data_path=outputs/prompts/$model/$dataset/$shots_list.jsonl
                test_list="test_${dataset2testsize[$dataset]}"
                val_list="val_${dataset2trainsize[$dataset]}_${val_prop}_$num_seed"

                # Predictions directories and lists
                base_output_dir="outputs/adaptation/$model/${model_type}_few_shot/$shots_list"
                cal_dir="outputs/adaptation/$model/${model_type}_few_shot_plus_dp_cal/${dataset}_${dataset2trainsize[$dataset]}_${val_prop}_$num_seed"
                prediction_dir="$base_output_dir/test=$dataset/list=$val_list"
                prediction_list="$val_list"
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
                        --max_seq_length $inference_max_seq_len
                fi

                # Calibration directories
                if [ ! -f "$cal_dir/test=$dataset/list=$test_list/logits.csv" ]; then
                    mkdir -p $cal_dir/test=$dataset/list=$test_list $cal_dir/logs
                    python -m llmcal2.scripts.affine_calibration \
                        --output_dir $cal_dir/test=$dataset/list=$test_list \
                        --log_dir $cal_dir/logs \
                        --checkpoint_dir $cal_dir \
                        --train_logits $prediction_dir/logits.csv \
                        --train_labels $prediction_dir/labels.csv \
                        --predict_logits $base_output_dir/test=$dataset/list=$test_list/logits.csv \
                        --predict_labels $base_output_dir/test=$dataset/list=$test_list/labels.csv \
                        --method $method \
                        --learning_rate $learning_rate \
                        --tolerance $tolerance \
                        --max_ls $max_ls \
                        --seed $seed
                fi
            done
        done
    done
done