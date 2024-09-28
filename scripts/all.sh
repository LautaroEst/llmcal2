#! /bin/bash -ex
export CUDA_VISIBLE_DEVICES=1

model="tinyllama"
DATASETS="sst2 dbpedia 20newsgroups banking77"
# DATASETS="sst2 "

# Supported models
declare -A model2checkpoint=(
    ["tinyllama"]="$LIT_CHECKPOINTS/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    ["phi3"]="$LIT_CHECKPOINTS/microsoft/Phi-3-mini-4k-instruct"
    ["llama3"]="$LIT_CHECKPOINTS/meta-llama/Meta-Llama-3-8B"
)

# Number of classes in each dataset
declare -A dataset2numclasses=(
    ["sst2"]=2
    ["agnews"]=4
    ["dbpedia"]=14
    ["20newsgroups"]=20
    ["banking77"]=77
)

# Number of samples to use for adaptation
if [ $model == "tinyllama" ]; then
    declare -A dataset2samples=(
        ["sst2"]="8 32 512"
        ["agnews"]="4 16 256"
        ["dbpedia"]="2 8 128"
        ["20newsgroups"]="2 8 128"
        ["banking77"]="1 4 64"
    )
else
    declare -A dataset2samples=(
        ["sst2"]="8 512"
        ["agnews"]="4 256"
        ["dbpedia"]="2 128"
        ["20newsgroups"]="2 128"
        ["banking77"]="1 64"
    )
fi

# Seeds
declare -A dataset2seed=(
    # ["sst2_8"]="639 923 932 6391 9322"
    # ["sst2_32"]="1564 1738 1783 15641 17832"
    # ["sst2_512"]="111 121 767 890 999"
    # ["agnews_4"]="295 926 962 2951 9622"
    # ["agnews_16"]="738 564 783 5641 7832"
    # ["agnews_256"]="493 821 812 4931 8212"
    # ["dbpedia_2"]="435 927 972 4351 9722"
    # ["dbpedia_8"]="338 364 383 3641 3832"
    # ["dbpedia_128"]="129 131 543 878 909"
    # ["20newsgroups_2"]="435 927 972 4351 9722"
    # ["20newsgroups_8"]="338 364 383 3641 3832"
    # ["20newsgroups_128"]="129 131 543 878 909"
    # ["banking77_1"]="322 444 848 858 868"
    # ["banking77_4"]="295 926 962 2951 9622"
    # ["banking77_64"]="131 888 893 912 933"
    ["sst2_8"]="639 923 932"
    ["sst2_32"]="1564 1738 1783"
    ["sst2_512"]="111 121 767"
    ["agnews_4"]="295 926 962"
    ["agnews_16"]="738 564 783"
    ["agnews_256"]="493 821 812"
    ["dbpedia_2"]="435 927 972"
    ["dbpedia_8"]="338 364 383"
    ["dbpedia_128"]="129 131 543"
    ["20newsgroups_2"]="435 927 972"
    ["20newsgroups_8"]="338 364 383"
    ["20newsgroups_128"]="129 131 543"
    ["banking77_1"]="322 444 848"
    ["banking77_4"]="295 926 962"
    ["banking77_64"]="131 888 893"
)

source ./scripts/prepare_datasets.sh
source ./scripts/prepare_prompts.sh
# source ./scripts/generative_no_adaptation.sh
# source ./scripts/affine_calibration.sh
source ./scripts/lora.sh
source ./scripts/lora_norm.sh
# source ./scripts/lora_no_es.sh
# ./scripts/lora_plus_affine_cal.sh
# ./scripts/lora_plus_affine_cal_no_es.sh
# ./scripts/lora_no_es.sh
