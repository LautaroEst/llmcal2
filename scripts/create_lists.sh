#!/bin/bash -ex

source ./scripts/env.sh

val_prop=0.3
for dataset in "${DATASETS[@]}"; do
    list_dir=lists/$dataset
    data_dir=data/$dataset
    total_train_size=${dataset2trainsize[$dataset]}
    test_size=${dataset2testsize[$dataset]}

    mkdir -p $list_dir $data_dir
    python -m llmcal2.scripts.create_lists --dataset $dataset --lists_dir $list_dir --data_dir $data_dir
    python -m llmcal2.scripts.create_lists --dataset $dataset --lists_dir $list_dir --data_dir $data_dir \
        --total_train_size $total_train_size --val_prop $val_prop --repetitions $num_seeds --seed $base_seed
    python -m llmcal2.scripts.create_lists --dataset $dataset --lists_dir $list_dir --data_dir $data_dir \
        --test_size $test_size --seed $base_seed
done

