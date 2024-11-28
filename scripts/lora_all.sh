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

for num_seed in $(seq 0 $((num_seeds-1))); do
    for loss in ${losses_options[@]}; do
        
        # Seed
        seed=$((base_seed+num_seed))

        # Lists
        train_data_paths=""
        train_lists=""
        val_lists=""
        list_identifier=""
        for d in ${DATASETS[@]}; do
            train_data_paths+="outputs/prompts/$model/$d/all.jsonl,"
            train_lists+="lists/$d/train_${dataset2trainsize[$d]}_${val_prop}_${num_seed}.txt,"
            val_lists+="lists/$d/val_${dataset2trainsize[$d]}_${val_prop}_${num_seed}.txt,"
            list_identifier+="${d}_${dataset2trainsize[$d]}_${val_prop}_${num_seed}__"
        done
        train_data_paths=$(echo "$train_data_paths" | sed 's/,$//')
        train_lists=$(echo "$train_lists" | sed 's/,$//')
        val_lists=$(echo "$val_lists" | sed 's/,$//')
        list_identifier=$(echo "$list_identifier" | sed 's/__$//')
        
        # Train directories
        train_output_dir="outputs/adaptation/$model/lora_${loss}/$list_identifier"
        log_dir=$train_output_dir/logs
        output_checkpoint_dir=$train_output_dir/checkpoint

        # Model directory
        model_dir=$CHECKPOINTS_DIR/${model2checkpoint[$model]}

        # Train
        if [ ! -f $train_output_dir/train_args.yaml ]; then
            mkdir -p $train_output_dir $log_dir $output_checkpoint_dir
            for file in config.json generation_config.json model_config.yaml tokenizer.json tokenizer.model tokenizer_config.json; do
                if [ -f $model_dir/$file ]; then
                    cp $model_dir/$file $output_checkpoint_dir
                fi
            done
            ln -sf $(readlink -f $model_dir/lit_model.pth) $output_checkpoint_dir/lit_model.pth
            python -m llmcal2.scripts.train_lora \
                --base_checkpoint_dir $model_dir \
                --data_paths $train_data_paths \
                --train_lists $train_lists \
                --val_lists $val_lists \
                --output_dir $train_output_dir \
                --output_checkpoint_dir $output_checkpoint_dir \
                --log_dir $log_dir \
                --precision $precision \
                --devices 1 \
                --num_nodes 1 \
                --global_batch_size $global_batch_size \
                --micro_batch_size $micro_batch_size \
                --val_check_interval 16 \
                --learning_rate $learning_rate \
                --optimizer $optimizer \
                --weight_decay $weight_decay \
                --loss $loss \
                --patience $patience \
                --seed $seed \
                $lora_args
        fi

        # Predict
        for dataset in ${DATASETS[@]}; do
            total_train_samples=${dataset2trainsize[$dataset]}
            cal_list="val_${total_train_samples}_${val_prop}_${num_seed}"
            test_list="test_${dataset2testsize[$dataset]}"
            predictions_dirs=("$train_output_dir/test=$dataset/list=$test_list" "$train_output_dir/test=$dataset/list=$cal_list")
            predictions_lists=("$test_list" "$cal_list")
            data_path="outputs/prompts/$model/$dataset/all.jsonl"
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
                        --max_seq_length $max_seq_length \
                        $lora_args
                fi
            done
        done
    done
done
