#!/bin/bash -ex

source ./scripts/env.sh
export CUDA_VISIBLE_DEVICES=1

val_prop=0.3

precision="32"
batch_size=32
learning_rate=1e-5
optimizer="adamw"
weight_decay=0.0
patience=10

declare -A model_dirs=(
    ["distilbert-base-uncased"]=$CHECKPOINTS_DIR/distilbert/distilbert-base-uncased
    # ["deberta-v2-xlarge"]=$CHECKPOINTS_DIR/microsoft/deberta-v2-xlarge
    ["roberta-large-mnli"]=$CHECKPOINTS_DIR/FacebookAI/roberta-large-mnli
)

for model in "${!model_dirs[@]}"; do
    if [ ! -d ${model_dirs[$model]} ]; then
        mkdir -p ${model_dirs[$model]}
        full_path=${model_dirs[$model]}
        model_url="${full_path#"$CHECKPOINTS_DIR/"}"
        huggingface-cli download $model_url --local-dir ${model_dirs[$model]} --include "*.json" "*.bin" "spm.model" --token $HF_TOKEN
    fi
    for dataset in ${DATASETS[@]}; do
        for num_seed in $(seq 0 $((num_seeds-1))); do
            # Seed
            seed=$((base_seed+num_seed))
            
            # Total train samples
            total_train_samples=${dataset2trainsize[$dataset]}

            # Lists
            train_list="train_${total_train_samples}_${val_prop}_$num_seed"
            val_list="val_${total_train_samples}_${val_prop}_$num_seed"
            test_list="test_${dataset2testsize[$dataset]}"

            # Train directories
            train_output_dir="outputs/adaptation/encoder_models/$model/${dataset}_${total_train_samples}_${val_prop}_$num_seed"
            log_dir=$train_output_dir/logs
            output_checkpoint_dir=$train_output_dir/checkpoint

            # Data paths
            data_path=data/$dataset/all.csv

            # Model directory
            model_dir=${model_dirs[$model]}

            # Train
            if [ ! -f "$train_output_dir/test=$dataset/list=$test_list/logits.csv" ]; then
                mkdir -p $train_output_dir/test=$dataset/list=$test_list $log_dir $output_checkpoint_dir
                for file in config.json tokenizer.json tokenizer.model tokenizer_config.json; do
                    if [ -f $model_dir/$file ]; then
                        cp $model_dir/$file $output_checkpoint_dir
                    fi
                done
                python -m llmcal2.scripts.train_encoder \
                    --base_checkpoint_dir $model_dir \
                    --data_path $data_path \
                    --train_list lists/$dataset/$train_list.txt \
                    --val_list lists/$dataset/$val_list.txt \
                    --test_list lists/$dataset/$test_list.txt \
                    --output_dir $train_output_dir \
                    --predictions_dir $train_output_dir/test=$dataset/list=$test_list \
                    --output_checkpoint_dir $output_checkpoint_dir \
                    --log_dir $log_dir \
                    --precision $precision \
                    --devices 1 \
                    --num_nodes 1 \
                    --global_batch_size $batch_size \
                    --micro_batch_size 4 \
                    --train_save_interval 8 \
                    --val_check_interval 16 \
                    --learning_rate $learning_rate \
                    --optimizer $optimizer \
                    --weight_decay $weight_decay \
                    --patience $patience \
                    --seed $seed
            fi
        done
    done
done