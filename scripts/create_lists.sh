#!/bin/bash -ex

### DISCLAIMER: It may happend that lists are generated differntly for different machines.
### Just use the ones provided in the lists directory.


declare -a DATASETS=("sst2" "agnews" "dbpedia" "20newsgroups" "banking77")

# Number of classes in each dataset
declare -A dataset2numclasses=(
    ["sst2"]=2
    ["agnews"]=4
    ["dbpedia"]=14
    ["20newsgroups"]=20
    ["banking77"]=77
)

# Number of samples to use for adaptation
declare -A dataset2samples=(
    ["sst2"]="8 32 512"
    ["agnews"]="4 16 256"
    ["dbpedia"]="2 8 128"
    ["20newsgroups"]="2 8 128"
    ["banking77"]="1 4 64"
)

# Seeds
declare -A dataset2seed=(
    ["sst2_8"]="639 923 932 6391 9322"
    ["sst2_32"]="1564 1738 1783 15641 17832"
    ["sst2_512"]="111 121 767 890 999"
    ["agnews_4"]="295 926 962 2951 9622"
    ["agnews_16"]="738 564 783 5641 7832"
    ["agnews_256"]="493 821 812 4931 8212"
    ["dbpedia_2"]="435 927 972 4351 9722"
    ["dbpedia_8"]="338 364 383 3641 3832"
    ["dbpedia_128"]="129 131 543 878 909"
    ["20newsgroups_2"]="435 927 972 4351 9722"
    ["20newsgroups_8"]="338 364 383 3641 3832"
    ["20newsgroups_128"]="129 131 543 878 909"
    ["banking77_1"]="322 444 848 858 868"
    ["banking77_4"]="295 926 962 2951 9622"
    ["banking77_64"]="131 888 893 912 933"
)

for dataset in ${DATASETS[@]}; do
    mkdir -p lists/$dataset
    python -m llmcal2.scripts.create_lists_all \
        --dataset $dataset \
        --output_dir lists/$dataset
    for size in ${dataset2samples[$dataset]}; do
        for random_state in ${dataset2seed[$dataset"_"$size]}; do
            total_train_samples=$((size * dataset2numclasses[$dataset]))
            use_train_samples_as_val=$((4 * dataset2numclasses[$dataset]))
            python -m llmcal2.scripts.create_lists \
                --data_dir outputs/prompts/generative/$dataset \
                --total_train_samples $total_train_samples \
                --val_prop 0.0 \
                --use_train_samples_as_val $use_train_samples_as_val \
                --random_state $random_state \
                --output_dir lists/$dataset
            python -m llmcal2.scripts.create_lists \
                --data_dir outputs/prompts/generative/$dataset \
                --total_train_samples $total_train_samples \
                --val_prop 0.3 \
                --use_train_samples_as_val -1 \
                --random_state $random_state \
                --output_dir lists/$dataset
        done
    done
done