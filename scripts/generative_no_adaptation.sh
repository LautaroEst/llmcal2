#! /bin/bash -ex
accelerator="gpu"
precision="bf16-true"
strategy="auto"
devices=1
num_nodes=1

checkpoint=${model2checkpoint[$model]}
output_dir="outputs/adaptation/$model/no_adaptation/$dataset/size=all/rs=all"
test_list=${dataset2testlist[$dataset]}
if [ ! -f $output_dir/test_logits.csv ]; then
    mkdir -p $output_dir
    python -m llmcal2.scripts.no_adaptation \
        --data_dir outputs/prompts/generative/$dataset \
        --checkpoint_dir $checkpoint \
        --test_list lists/$dataset/$test_list.txt \
        --batch_size 1 \
        --accelerator $accelerator \
        --strategy $strategy \
        --devices $devices \
        --num_nodes $num_nodes \
        --precision $precision \
        --output_dir $output_dir
fi
for size in ${dataset2samples[$dataset]}; do
    for random_state in ${dataset2seed[$dataset"_"$size]}; do
        total_train_samples=$((size * dataset2numclasses[$dataset]))
        output_dir="outputs/adaptation/$model/no_adaptation/$dataset/size=$size/rs=$random_state"
        if [ ! -f $output_dir/train_logits.csv ]; then
            mkdir -p $output_dir
            python -m llmcal2.scripts.no_adaptation \
                --data_dir outputs/prompts/generative/$dataset \
                --checkpoint_dir $checkpoint \
                --train_list lists/$dataset/train--total_train_samples=${total_train_samples}_val_prop=0.3_random_state=${random_state}.txt \
                --val_list lists/$dataset/val--total_train_samples=${total_train_samples}_val_prop=0.3_random_state=${random_state}.txt \
                --batch_size 1 \
                --accelerator $accelerator \
                --strategy $strategy \
                --devices $devices \
                --num_nodes $num_nodes \
                --precision $precision \
                --output_dir $output_dir
        fi
    done
done