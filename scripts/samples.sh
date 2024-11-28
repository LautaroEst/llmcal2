#!/bin/bash

source ./scripts/env.sh

precision="bf16-true"

declare -A dataset2trainsizes=(
    ["sst2"]="1024 256 512"
    ["agnews"]="1024 256 512"
    ["dbpedia"]="1792 448 896"
    ["20newsgroups"]="2560 640 1280"
    ["banking77"]="4928 1232 2464"
)

declare -A model_dirs=(
    ["no_adaptation"]=$CHECKPOINTS_DIR/${model2checkpoint[$model]}
    ["instruct"]=$CHECKPOINTS_DIR/${model2checkpoint[${model}-instruct]}
)

val_prop=0.3

model_type="instruct"
model_dir=${model_dirs[$model_type]}

# $1: dataset
# $2: size
# $3: num_seed
# $4: method
run_lora() {
    local data_path="outputs/prompts/$model/$1/all.jsonl"
    if [ $4 == "lora_ans_all_train" ] || [ $4 == "lora_ans_no_es" ]; then
        local train_list="train_${2}_0.0_${3}"
        local train_output_dir="outputs/adaptation/$model/$4/${1}_${2}_0.0_${3}"
        if [ $4 == "lora_ans_all_train" ]; then
            local max_steps=$(python -c "import torch; print(torch.load('outputs/adaptation/$model/lora_ans/${1}_${2}_0.3_${3}/best.ckpt', weights_only=False)['step_count'],end='')")
        else
            local max_steps=-1
        fi
    else
        local train_list="train_${2}_0.3_${3}"
        local train_output_dir="outputs/adaptation/$model/$4/${1}_${2}_0.3_${3}"
        local max_steps=-1
    fi
    local val_list="val_${2}_0.3_${3}"
    local test_list="test_${dataset2testsize[$1]}"
    local output_checkpoint_dir=$train_output_dir/checkpoint
    local log_dir=$train_output_dir/logs
    local output_dir="$train_output_dir/test=$1/list=$test_list"
    local seed=$((base_seed+$3))
    local lora_args="--lora_r=8 --lora_alpha=16 --lora_dropout=0.05 --lora_query --lora_key --lora_value --lora_projection --lora_mlp --lora_head"
    local val_prop=0.0
    local global_batch_size=8
    local micro_batch_size=1
    local learning_rate=0.0001
    local optimizer="adamw"
    local weight_decay=0.0
    local patience=10

    if [ ! -f $train_output_dir/train_args.yaml ]; then
        echo "Running $4 for $1 with $2 samples and seed $seed in $train_output_dir"
        mkdir -p $train_output_dir $log_dir $output_checkpoint_dir
        for file in config.json generation_config.json model_config.yaml tokenizer.json tokenizer.model tokenizer_config.json; do
            if [ -f $model_dir/$file ]; then
                cp $model_dir/$file $output_checkpoint_dir
            fi
        done
        ln -sf $(readlink -f $model_dir/lit_model.pth) $output_checkpoint_dir/lit_model.pth
        python -m llmcal2.scripts.train_lora \
            --base_checkpoint_dir $model_dir \
            --data_paths $data_path \
            --train_lists lists/$dataset/$train_list.txt \
            --val_lists lists/$dataset/$val_list.txt \
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
            --loss "ans" \
            --patience $patience \
            --max_steps $max_steps \
            --seed $seed \
            $lora_args
    fi

    if [ ! -f $output_dir/logits.csv ]; then
        echo "Running prediction for $1 with $2 samples and seed $seed in $output_dir"
        mkdir -p $output_dir
        python -m llmcal2.scripts.run_posteriors \
            --checkpoint_dir $output_checkpoint_dir \
            --peft "lora" \
            --data_path $data_path \
            --output_dir $prediction_dir \
            --prediction_lists lists/$dataset/$test_list.txt \
            --precision $precision \
            --devices 1 \
            --num_nodes 1 \
            --batch_size 1 \
            --max_seq_length $max_seq_length \
            $lora_args
    fi
}

run_lora_label_smoothing() {
    local data_path="outputs/prompts/$model/$1/all.jsonl"
    local train_list="val_${2}_0.3_${3}"
    local train_output_dir="outputs/adaptation/$model/lora_label_smoothing/${1}_${2}_0.3_${3}"
    local max_steps=-1
    local val_list="val_${2}_0.3_${3}"
    local test_list="test_${dataset2testsize[$1]}"
    local output_checkpoint_dir=$train_output_dir/checkpoint
    local log_dir=$train_output_dir/logs
    local output_dir="$train_output_dir/test=$1/list=$test_list"
    local seed=$((base_seed+$3))
    local lora_args="--lora_r=8 --lora_alpha=16 --lora_dropout=0.05 --lora_query --lora_key --lora_value --lora_projection --lora_mlp --lora_head"
    local val_prop=0.0
    local global_batch_size=8
    local micro_batch_size=1
    local learning_rate=0.0001
    local optimizer="adamw"
    local weight_decay=0.0
    local patience=10

    local cal_logits_dir="outputs/adaptation/$model/${model_type}_plus_dp_cal/${1}_${2}_0.3_${3}/test=$1/list=$val_list"
    if [ ! -f $cal_logits_dir/logits.csv ]; then
        mkdir -p $cal_logits_dir
        python -m llmcal2.scripts.affine_prediction \
            --checkpoint_path $cal_logits_dir/../../last.ckpt \
            --method "dp_calibration" \
            --predict_logits outputs/adaptation/$model/${model_type}/${1}_${2}_0.3_${3}/test=$1/list=$val_list/logits.csv \
            --predict_labels outputs/adaptation/$model/${model_type}/${1}_${2}_0.3_${3}/test=$1/list=$val_list/labels.csv \
            --output_dir $cal_logits_dir
    fi

    if [ ! -f $train_output_dir/train_args.yaml ]; then
        echo "Running lora_label_smoothing for $1 with $2 samples and seed $seed in $train_output_dir"
        mkdir -p $train_output_dir $log_dir $output_checkpoint_dir
        for file in config.json generation_config.json model_config.yaml tokenizer.json tokenizer.model tokenizer_config.json; do
            if [ -f $model_dir/$file ]; then
                cp $model_dir/$file $output_checkpoint_dir
            fi
        done
        ln -sf $(readlink -f $model_dir/lit_model.pth) $output_checkpoint_dir/lit_model.pth
        python -m llmcal2.scripts.train_lora_label_smoothing \
            --base_checkpoint_dir $model_dir \
            --data_paths $data_path \
            --train_lists lists/$dataset/$train_list.txt \
            --train_logits $cal_logits_dir/logits.csv \
            --train_labels $cal_logits_dir/labels.csv \
            --val_lists lists/$dataset/$val_list.txt \
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
            --loss "ans" \
            --patience $patience \
            --max_steps $max_steps \
            --seed $seed \
            $lora_args
    fi

    if [ ! -f $output_dir/logits.csv ]; then
        echo "Running prediction for $1 with $2 samples and seed $seed in $output_dir"
        mkdir -p $output_dir
        python -m llmcal2.scripts.run_posteriors \
            --checkpoint_dir $output_checkpoint_dir \
            --peft "lora" \
            --data_path $data_path \
            --output_dir $prediction_dir \
            --prediction_lists lists/$dataset/$test_list.txt \
            --precision $precision \
            --devices 1 \
            --num_nodes 1 \
            --batch_size 1 \
            --max_seq_length $max_seq_length \
            $lora_args
    fi
}

run_calibration() {
    local data_path="outputs/prompts/$model/$1/all.jsonl"
    local train_list="train_${2}_0.3_${3}"
    local val_list="val_${2}_0.3_${3}"
    local test_list="test_${dataset2testsize[$1]}"
    local prediction_dir="outputs/adaptation/$model/$model_type/all/test=$1/list=$test_list"
    local train_output_dir="outputs/adaptation/$model/${model_type}_plus_dp_cal/${1}_${2}_0.3_${3}"
    local output_dir="$train_output_dir/test=$1/list=$test_list"
    local seed=$((base_seed+$3))
    if [ ! -f $prediction_dir/logits.csv ]; then
        echo "Running predictions for calibration for $1 with $2 samples and seed $seed in $prediction_dir"
        mkdir -p $prediction_dir
        python -m llmcal2.scripts.run_posteriors \
            --checkpoint_dir $model_dir \
            --data_path $data_path \
            --output_dir $prediction_dir \
            --prediction_lists lists/$1/$train_list.txt \
            --precision $precision \
            --devices 1 \
            --num_nodes 1 \
            --batch_size 1 \
            --max_seq_length $max_seq_length
        mkdir -p $output_dir $train_output_dir/logs
    fi
    if [ ! -f $output_dir/logits.csv ]; then
        echo "Running calibration for $1 with $2 samples and seed $seed in $output_dir"
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
    fi
}

run_calibration_in_lora_model() {
    local data_path="outputs/prompts/$model/$1/all.jsonl"
    local train_list="train_${2}_0.3_${3}"
    local val_list="val_${2}_0.3_${3}"
    local test_list="test_${dataset2testsize[$1]}"
    local prediction_dir="outputs/adaptation/$model/lora_ans/${1}_${2}_0.3_${3}/test=$1/list=$val_list"
    local train_output_dir="outputs/adaptation/$model/lora_ans_plus_dp_cal/${1}_${2}_0.3_${3}"
    local output_dir="$train_output_dir/test=$1/list=$test_list"
    local seed=$((base_seed+$3))
    if [ ! -f $prediction_dir/logits.csv ]; then
        echo "Running predictions for calibration for $1 with $2 samples and seed $seed in $prediction_dir"
        mkdir -p $prediction_dir
        python -m llmcal2.scripts.run_posteriors \
            --checkpoint_dir $model_dir \
            --data_path $data_path \
            --peft "lora" \
            --output_dir $prediction_dir \
            --prediction_lists lists/$1/$train_list.txt \
            --precision $precision \
            --devices 1 \
            --num_nodes 1 \
            --batch_size 1 \
            --max_seq_length $max_seq_length
        mkdir -p $output_dir $train_output_dir/logs
    fi
    if [ ! -f $output_dir/logits.csv ]; then
        echo "Running calibration for $1 with $2 samples and seed $seed in $output_dir"
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
    fi
}

for dataset in "${!dataset2trainsizes[@]}"; do
    for size in ${dataset2trainsizes[$dataset]}; do
        for num_seed in $(seq 0 $((num_seeds-1))); do
            for method in ${model_type}_plus_dp_cal lora_ans_plus_dp_cal lora_ans lora_ans_no_es lora_ans_all_train lora_label_smoothing; do
                if [ $method == "${model_type}_plus_dp_cal" ]; then
                    run_calibration $dataset $size $num_seed
                elif [ $method == "lora_ans" ] || [ $method == "lora_ans_no_es" ] || [ $method == "lora_ans_all_train" ]; then
                    run_lora $dataset $size $num_seed $method
                elif [ $method == "lora_ans_plus_dp_cal" ]; then
                    run_lora $dataset $size $num_seed "lora_ans"
                    run_calibration_in_lora_model $dataset $size $num_seed
                elif [ $method == "lora_label_smoothing" ]; then
                    run_lora_label_smoothing $dataset $size $num_seed
                fi
            done
        done
    done
done
            