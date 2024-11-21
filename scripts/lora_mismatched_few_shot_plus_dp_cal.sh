#!/bin/bash -ex

source ./scripts/env.sh

lora_args="--lora_r=8 --lora_alpha=16 --lora_dropout=0.05 --lora_query --lora_key --lora_value --lora_projection --lora_mlp --lora_head"
val_prop=0.3
method="dp_calibration"
learning_rate=1e-3
tolerance=1e-5
max_ls=40

for dataset in ${DATASETS[@]}; do
    for n_shot in "${N_SHOTS[@]}"; do
        for num_seed in $(seq 0 $((num_seeds-1))); do
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
                list_identifier+="__${n_shot}shots_${num_seed}"
                cal_list="val_${total_train_samples}_${val_prop}_${num_seed}"
                test_list="test_${dataset2testsize[$dataset]}"
                
                # Train directories
                cal_logits_dir="outputs/adaptation/$model/lora_${loss}_few_shot/$list_identifier"
                train_output_dir="outputs/adaptation/$model/lora_${loss}_few_shot_plus_dp_cal/$list_identifier"
                log_dir=$train_output_dir/logs

                # Train calibration on val set and predict on test set
                if [ ! -f "$train_output_dir/test=$dataset/list=$test_list/logits.csv" ]; then
                    mkdir -p $train_output_dir/test=$dataset/list=$test_list $log_dir
                    python -m llmcal2.scripts.affine_calibration \
                        --output_dir $train_output_dir/test=$dataset/list=$test_list \
                        --log_dir $log_dir \
                        --checkpoint_dir $train_output_dir \
                        --train_logits $cal_logits_dir/test=$dataset/list=$cal_list/logits.csv \
                        --train_labels $cal_logits_dir/test=$dataset/list=$cal_list/labels.csv \
                        --predict_logits $cal_logits_dir/test=$dataset/list=$test_list/logits.csv \
                        --predict_labels $cal_logits_dir/test=$dataset/list=$test_list/labels.csv \
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
