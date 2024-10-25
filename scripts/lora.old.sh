#!/bin/bash -ex

source ./scripts/env.sh

lora_args="--lora_r=8 --lora_alpha=16 --lora_dropout=0.05 --lora_query --lora_key --lora_value --lora_projection --lora_mlp --lora_head"

for train_dataset in ${DATASETS[@]}; do
    train_data_path=outputs/prompts/$model/$train_dataset/all.jsonl

    for test_dataset in ${DATASETS[@]}; do
        test_data_path=outputs/prompts/$model/$test_dataset/all.jsonl
        test_list="test_${dataset2testsize[$test_dataset]}"

        for num_seed in $(seq 0 $((num_seeds-1))); do
            seed=$((base_seed+num_seed))

            total_size=${dataset2trainsize[$train_dataset]}
            train_size=$((total_size*7/10))
            val_size=$((total_size-train_size))

            train_list="train_${train_size}_$num_seed"
            val_list="val_${val_size}_$num_seed"
            
            for loss in fs ans norm-5; do

                train_output_dir="outputs/adaptation/$model/lora_${loss}/train=$train_dataset/list=$train_list"
                log_dir=$train_output_dir/logs
                output_checkpoint_dir=$train_output_dir/checkpoint
                test_output_dir="$train_output_dir/test=$test_dataset/list=$test_list"

                if [ ! -f $train_output_dir/train_args.yaml ]; then
                    mkdir -p $train_output_dir $log_dir $output_checkpoint_dir
                    for file in config.json generation_config.json model_config.yaml tokenizer.json tokenizer.model tokenizer_config.json; do
                        if [ -f $CHECKPOINTS_DIR/${model2checkpoint[$model]}/$file ]; then
                            cp $CHECKPOINTS_DIR/${model2checkpoint[$model]}/$file $output_checkpoint_dir
                        fi
                    done
                    ln -sf $(readlink -f $CHECKPOINTS_DIR/${model2checkpoint[$model]}/lit_model.pth) $output_checkpoint_dir/lit_model.pth
                    python -m llmcal2.scripts.train_lora \
                        --base_checkpoint_dir $CHECKPOINTS_DIR/${model2checkpoint[$model]} \
                        --data_path $train_data_path \
                        --train_list lists/$train_dataset/$train_list.txt \
                        --val_list lists/$train_dataset/$val_list.txt \
                        --output_dir $train_output_dir \
                        --output_checkpoint_dir $output_checkpoint_dir \
                        --log_dir $log_dir \
                        --precision "bf16-true" \
                        --devices 1 \
                        --num_nodes 1 \
                        --global_batch_size 8 \
                        --micro_batch_size 1 \
                        --val_check_interval 4 \
                        --learning_rate 0.0001 \
                        --optimizer "adamw" \
                        --weight_decay 0.0 \
                        --loss $loss \
                        --patience 10 \
                        --seed $seed \
                        $lora_args
                fi

                if [ ! -f $test_output_dir/logits.csv ]; then
                    mkdir -p $test_output_dir
                    python -m llmcal2.scripts.run_posteriors \
                        --checkpoint_dir $output_checkpoint_dir \
                        --peft "lora" \
                        --data_path $test_data_path \
                        --output_dir $test_output_dir \
                        --prediction_list lists/$test_dataset/$test_list.txt \
                        --precision "bf16-true" \
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