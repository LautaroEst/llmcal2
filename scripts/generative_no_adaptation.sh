#! /bin/bash -ex
accelerator="gpu"
precision="bf16-true"
strategy="auto"
devices=1
num_nodes=1

checkpoint=${model2checkpoint[$model]}
for dataset in $DATASETS; do
    output_dir="outputs/adaptation/$model/no_adaptation/$dataset/size=all/rs=all"
    if [ ! -f $output_dir/test_logits.csv ]; then
        mkdir -p $output_dir
        python -m llmcal2.scripts.no_adaptation \
            --data_dir outputs/prompts/generative/$dataset \
            --total_train_samples 10 \
            --val_prop 0.3 \
            --random_state 0 \
            --checkpoint_dir $checkpoint \
            --batch_size 1 \
            --accelerator $accelerator \
            --strategy $strategy \
            --devices $devices \
            --num_nodes $num_nodes \
            --precision $precision \
            --test_output_dir $output_dir
    fi
    for size in ${dataset2samples[$dataset]}; do
        for random_state in ${dataset2seed[$dataset"_"$size]}; do
            total_train_samples=$((size * dataset2numclasses[$dataset]))
            output_dir="outputs/adaptation/$model/no_adaptation/$dataset/size=$size/rs=$random_state"
            if [ ! -f $output_dir/train_logits.csv ]; then
                mkdir -p $output_dir
                python -m llmcal2.scripts.no_adaptation \
                    --data_dir outputs/prompts/generative/$dataset \
                    --total_train_samples $total_train_samples \
                    --val_prop 0.3 \
                    --random_state $random_state \
                    --checkpoint_dir $checkpoint \
                    --batch_size 1 \
                    --accelerator $accelerator \
                    --strategy $strategy \
                    --devices $devices \
                    --num_nodes $num_nodes \
                    --precision $precision \
                    --trainval_output_dir $output_dir
            fi
        done
    done
done