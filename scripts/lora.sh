#!/bin/bash -ex

source ./scripts/env.sh

lora_args="--lora_r=8 --lora_alpha=16 --lora_dropout=0.05 --lora_query --lora_key --lora_value --lora_projection --lora_mlp --lora_head"
val_prop=0.3
for dataset in ${DATASETS[@]}; do
    test_data_path=outputs/prompts/$model/$dataset/all.jsonl
    test_list="test_${dataset2testsize[$dataset]}"
    # Repeat for each seed
    for num_seed in $(seq 0 $((num_seeds-1))); do
        seed=$((base_seed+num_seed))
        total_size=${dataset2trainsize[$dataset]}

        #### Train LORA on one dataset and test on same dataset
        train_data_paths=outputs/prompts/$model/$dataset/all.jsonl
        train_list="lists/$dataset/train_${total_size}_${val_prop}_$num_seed.txt"
        val_list="lists/$dataset/val_${total_size}_${val_prop}_$num_seed.txt"
        list_identifier="${dataset}_${total_size}_${val_prop}_$num_seed"
        # Do for each loss
        for loss in $losses_options; do
            train_output_dir="outputs/adaptation/$model/lora_${loss}/$list_identifier"
            log_dir=$train_output_dir/logs
            output_checkpoint_dir=$train_output_dir/checkpoint
            test_output_dir="$train_output_dir/test=$dataset/list=$test_list"
            # Train
            if [ ! -f $train_output_dir/train_args.yaml ]; then
                mkdir -p $train_output_dir $log_dir $output_checkpoint_dir
                for file in config.json generation_config.json model_config.yaml tokenizer.json tokenizer.model tokenizer_config.json; do
                    if [ -f $CHECKPOINTS_DIR/${model2checkpoint[$model]}/$file ]; then
                        cp $CHECKPOINTS_DIR/${model2checkpoint[$model]}/$file $output_checkpoint_dir
                    fi
                done
                ln -sf $(readlink -f $CHECKPOINTS_DIR/${model2checkpoint[$model]}/lit_model.pth) $output_checkpoint_dir/lit_model.pth
                export CUDA_VISIBLE_DEVICES=1
                python -m llmcal2.scripts.train_lora \
                    --base_checkpoint_dir $CHECKPOINTS_DIR/${model2checkpoint[$model]} \
                    --data_paths $train_data_paths \
                    --train_lists $train_list \
                    --val_lists $val_list \
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
            # Predict
            if [ ! -f $test_output_dir/logits.csv ]; then
                mkdir -p $test_output_dir
                export CUDA_VISIBLE_DEVICES=0,1
                python -m llmcal2.scripts.run_posteriors \
                    --checkpoint_dir $output_checkpoint_dir \
                    --peft "lora" \
                    --data_path $test_data_path \
                    --output_dir $test_output_dir \
                    --prediction_lists lists/$dataset/$test_list.txt \
                    --precision "bf16-true" \
                    --devices 2 \
                    --num_nodes 1 \
                    --batch_size 1 \
                    --max_seq_length $max_seq_length \
                    $lora_args
            fi
        done

        #### Train LORA on all datasets but one and test on remaining dataset
        train_data_paths=""
        train_lists=""
        val_lists=""
        list_identifier=""
        for d in ${DATASETS[@]}; do
            if [ $d != $dataset ]; then
                train_data_paths+="outputs/prompts/$model/$d/all.jsonl,"
                train_lists+="lists/$d/train_${dataset2trainsize[$d]}_${val_prop}_${num_seed}.txt,"
                val_lists+="lists/$d/val_${dataset2trainsize[$d]}_${val_prop}_${num_seed}.txt,"
                list_identifier+="${d}_${dataset2trainsize[$d]}_${val_prop}_${num_seed}__"
            fi
        done
        train_lists=$(echo "$train_lists" | sed 's/,$//')
        val_lists=$(echo "$val_lists" | sed 's/,$//')
        list_identifier=$(echo "$list_identifier" | sed 's/__$//')
        # Do for each loss
        for loss in $losses_options; do
            train_output_dir="outputs/adaptation/$model/lora_${loss}/$list_identifier"
            log_dir=$train_output_dir/logs
            output_checkpoint_dir=$train_output_dir/checkpoint
            test_output_dir="$train_output_dir/test=$dataset/list=$test_list"
            # Train
            if [ ! -f $train_output_dir/train_args.yaml ]; then
                mkdir -p $train_output_dir $log_dir $output_checkpoint_dir
                for file in config.json generation_config.json model_config.yaml tokenizer.json tokenizer.model tokenizer_config.json; do
                    if [ -f $CHECKPOINTS_DIR/${model2checkpoint[$model]}/$file ]; then
                        cp $CHECKPOINTS_DIR/${model2checkpoint[$model]}/$file $output_checkpoint_dir
                    fi
                done
                ln -sf $(readlink -f $CHECKPOINTS_DIR/${model2checkpoint[$model]}/lit_model.pth) $output_checkpoint_dir/lit_model.pth
                export CUDA_VISIBLE_DEVICES=1
                python -m llmcal2.scripts.train_lora \
                    --base_checkpoint_dir $CHECKPOINTS_DIR/${model2checkpoint[$model]} \
                    --data_paths $train_data_paths \
                    --train_lists $train_lists \
                    --val_lists $val_lists \
                    --output_dir $train_output_dir \
                    --output_checkpoint_dir $output_checkpoint_dir \
                    --log_dir $log_dir \
                    --precision "bf16-true" \
                    --devices 1 \
                    --num_nodes 1 \
                    --global_batch_size 8 \
                    --micro_batch_size 1 \
                    --val_check_interval 16 \
                    --learning_rate 0.0001 \
                    --optimizer "adamw" \
                    --weight_decay 0.0 \
                    --loss $loss \
                    --patience 10 \
                    --seed $seed \
                    $lora_args
            fi
            # Predict
            if [ ! -f $test_output_dir/logits.csv ]; then
                mkdir -p $test_output_dir
                export CUDA_VISIBLE_DEVICES=0,1
                python -m llmcal2.scripts.run_posteriors \
                    --checkpoint_dir $output_checkpoint_dir \
                    --peft "lora" \
                    --data_path $test_data_path \
                    --output_dir $test_output_dir \
                    --prediction_lists lists/$dataset/$test_list.txt \
                    --precision "bf16-true" \
                    --devices 2 \
                    --num_nodes 1 \
                    --batch_size 1 \
                    --max_seq_length $max_seq_length \
                    $lora_args
            fi
        done
    done
done


#### Train LORA on all datasets
# Repeat for each seed
for num_seed in $(seq 0 $((num_seeds-1))); do
    seed=$((base_seed+num_seed))

    # Do for each loss
    for loss in $losses_options; do

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
        train_lists=$(echo "$train_lists" | sed 's/,$//')
        val_lists=$(echo "$val_lists" | sed 's/,$//')
        list_identifier=$(echo "$list_identifier" | sed 's/__$//')
        
        # Train
        train_output_dir="outputs/adaptation/$model/lora_${loss}/$list_identifier"
        log_dir=$train_output_dir/logs
        output_checkpoint_dir=$train_output_dir/checkpoint
        if [ ! -f $train_output_dir/train_args.yaml ]; then
            mkdir -p $train_output_dir $log_dir $output_checkpoint_dir
            for file in config.json generation_config.json model_config.yaml tokenizer.json tokenizer.model tokenizer_config.json; do
                if [ -f $CHECKPOINTS_DIR/${model2checkpoint[$model]}/$file ]; then
                    cp $CHECKPOINTS_DIR/${model2checkpoint[$model]}/$file $output_checkpoint_dir
                fi
            done
            ln -sf $(readlink -f $CHECKPOINTS_DIR/${model2checkpoint[$model]}/lit_model.pth) $output_checkpoint_dir/lit_model.pth
            export CUDA_VISIBLE_DEVICES=1
            python -m llmcal2.scripts.train_lora \
                --base_checkpoint_dir $CHECKPOINTS_DIR/${model2checkpoint[$model]} \
                --data_paths $train_data_paths \
                --train_lists $train_lists \
                --val_lists $val_lists \
                --output_dir $train_output_dir \
                --output_checkpoint_dir $output_checkpoint_dir \
                --log_dir $log_dir \
                --precision "bf16-true" \
                --devices 1 \
                --num_nodes 1 \
                --global_batch_size 8 \
                --micro_batch_size 1 \
                --val_check_interval 16 \
                --learning_rate 0.0001 \
                --optimizer "adamw" \
                --weight_decay 0.0 \
                --loss $loss \
                --patience 10 \
                --seed $seed \
                $lora_args
        fi

        for dataset in ${DATASETS[@]}; do
            test_data_path=outputs/prompts/$model/$dataset/all.jsonl
            test_list="test_${dataset2testsize[$dataset]}"
            test_output_dir="$train_output_dir/test=$dataset/list=$test_list"
            # Predict
            if [ ! -f $test_output_dir/logits.csv ]; then
                mkdir -p $test_output_dir
                export CUDA_VISIBLE_DEVICES=0,1
                python -m llmcal2.scripts.run_posteriors \
                    --checkpoint_dir $output_checkpoint_dir \
                    --peft "lora" \
                    --data_path $test_data_path \
                    --output_dir $test_output_dir \
                    --prediction_lists lists/$dataset/$test_list.txt \
                    --precision "bf16-true" \
                    --devices 2 \
                    --num_nodes 1 \
                    --batch_size 1 \
                    --max_seq_length $max_seq_length \
                    $lora_args
            fi
        done
    done
done