#!/bin/bash -ex

source ./scripts/env.sh

for dataset in "${DATASETS[@]}"; do
    data_path=outputs/prompts/$model/$dataset/all.jsonl
    test_list="test_${dataset2testsize[$dataset]}"
    test_output_dir="outputs/adaptation/$model/no_adaptation/test=$dataset/list=$test_list"
    # Repeat for each seed
    for num_seed in $(seq 0 $((num_seeds-1))); do
        seed=$((base_seed+num_seed))
        total_size=${dataset2trainsize[$dataset]}
        list_identifier="${dataset}_${total_size}_${val_prop}_$num_seed"

        # Base version
        base_predictions_dir="outputs/adaptation/$model/dp_calibration/$list_identifier"
        predictions_list="val_${total_size}_${val_prop}_$num_seed"
        predictions_dir="$base_predictions_dir/uncal-$predictions_list"
        if [ ! -f $predictions_dir/logits.csv ]; then
            mkdir -p $predictions_dir
            export CUDA_VISIBLE_DEVICES=0,1
            python -m llmcal2.scripts.run_posteriors \
                --checkpoint_dir $CHECKPOINTS_DIR/${model2checkpoint[${model}]} \
                --data_path $data_path \
                --output_dir $predictions_dir \
                --prediction_lists lists/$dataset/$predictions_list.txt \
                --precision "bf16-true" \
                --devices 2 \
                --num_nodes 1 \
                --batch_size 1 \
                --max_seq_length $max_seq_length
        fi
        # Train calibration on val set and predict on test set
        cal_output_dir="$base_predictions_dir/test=$dataset/list=$test_list"
        cal_log_dir="$base_predictions_dir/logs"
        if [ ! -f $cal_output_dir/logits.csv ]; then
            mkdir -p $cal_output_dir
            python -m llmcal2.scripts.affine_calibration \
                --output_dir $cal_output_dir \
                --log_dir $cal_log_dir \
                --train_logits $predictions_dir/logits.csv \
                --train_labels $predictions_dir/labels.csv \
                --predict_logits $test_output_dir/logits.csv \
                --predict_labels $test_output_dir/labels.csv \
                --method "dp_calibration" \
                --learning_rate 1e-4 \
                --tolerance 1e-5 \
                --max_ls 100
        fi


        # Instructions version
        if [ ! -z ${model2checkpoint[${model}-instruct]} ]; then
            base_predictions_dir="outputs/adaptation/$model/instructions_dp_calibration/$list_identifier"
            predictions_list="val_${total_size}_${val_prop}_$num_seed"
            predictions_dir="$base_predictions_dir/uncal-$predictions_list"
            if [ ! -f $predictions_dir/logits.csv ]; then
                mkdir -p $predictions_dir
                export CUDA_VISIBLE_DEVICES=0,1
                python -m llmcal2.scripts.run_posteriors \
                    --checkpoint_dir $CHECKPOINTS_DIR/${model2checkpoint[${model}-instruct]} \
                    --data_path $data_path \
                    --output_dir $predictions_dir \
                    --prediction_lists lists/$dataset/$predictions_list.txt \
                    --precision "bf16-true" \
                    --devices 2 \
                    --num_nodes 1 \
                    --batch_size 1 \
                    --max_seq_length $max_seq_length
            fi
            # Train calibration on val set and predict on test set
            cal_output_dir="$base_predictions_dir/test=$dataset/list=$test_list"
            cal_log_dir="$base_predictions_dir/logs"
            if [ ! -f $cal_output_dir/logits.csv ]; then
                mkdir -p $cal_output_dir
                python -m llmcal2.scripts.affine_calibration \
                    --output_dir $cal_output_dir \
                    --log_dir $cal_log_dir \
                    --train_logits $predictions_dir/logits.csv \
                    --train_labels $predictions_dir/labels.csv \
                    --predict_logits $test_output_dir/logits.csv \
                    --predict_labels $test_output_dir/labels.csv \
                    --method "dp_calibration" \
                    --learning_rate 1e-4 \
                    --tolerance 1e-5 \
                    --max_ls 100
            fi
        fi
    done
done