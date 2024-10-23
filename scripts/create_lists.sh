#!/bin/bash -ex

source ./scripts/env.sh

declare -A dataset2trainsize=(
    ["sst2"]=1024
    ["agnews"]=1024
    ["dbpedia"]=1792
    ["20newsgroups"]=2560
    ["banking77"]=4928
)

declare -A dataset2testsize=(
    ["sst2"]=400
    ["agnews"]=400
    ["dbpedia"]=700
    ["20newsgroups"]=800
    ["banking77"]=1000
)

for dataset in "${DATASETS[@]}"; do
    list_dir=lists/$dataset
    data_dir=data/$dataset

    total_size=${dataset2trainsize[$dataset]}
    train_size=$((total_size*7/10))
    val_size=$((total_size-train_size))
    test_size=${dataset2testsize[$dataset]}

    mkdir -p $list_dir $data_dir
    python -m llmcal2.scripts.create_lists --dataset $dataset --lists_dir $list_dir --data_dir $data_dir \
        --train_size $train_size --val_size $val_size --test_size $test_size --repetitions $num_seeds --seed $base_seed
done