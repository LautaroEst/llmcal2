#!/bin/bash -ex

source ./scripts/env.sh

declare -A dataset2trainsizes=(
    ["sst2"]="16 64 128 256 512 1024"
    ["agnews"]="16 64 128 256 512 1024"
    ["dbpedia"]="28 112 224 448 896 1792"
    ["20newsgroups"]="40 160 320 640 1280 2560"
    ["banking77"]="77 308 616 1232 2464 4928"
)

val_prop=0.3
for dataset in "${DATASETS[@]}"; do
    list_dir=lists/$dataset
    data_dir=data/$dataset
    total_train_sizes=${dataset2trainsizes[$dataset]}
    test_size=${dataset2testsize[$dataset]}
    for total_train_size in ${total_train_sizes}; do
        mkdir -p $list_dir $data_dir
        python -m llmcal2.scripts.create_lists --dataset $dataset --lists_dir $list_dir --data_dir $data_dir
        python -m llmcal2.scripts.create_lists --dataset $dataset --lists_dir $list_dir --data_dir $data_dir \
            --total_train_size $total_train_size --val_prop $val_prop --repetitions $num_seeds --seed $base_seed
        python -m llmcal2.scripts.create_lists --dataset $dataset --lists_dir $list_dir --data_dir $data_dir \
            --test_size $test_size --seed $base_seed

        for num_seed in $(seq 0 $((num_seeds-1))); do
            cat $list_dir/train_${total_train_size}_0.3_$num_seed.txt $list_dir/val_${total_train_size}_0.3_$num_seed.txt > $list_dir/train_${total_train_size}_0.0_$num_seed.txt
            for n_shots in 2 4 8 16 32; do
                shots_list=${n_shots}shots_${num_seed}
                head -n $n_shots $list_dir/val_${total_train_size}_${val_prop}_${num_seed}.txt > $list_dir/$shots_list.txt
            done
        done
    done
done

