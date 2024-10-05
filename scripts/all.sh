#! /bin/bash -ex
export CUDA_VISIBLE_DEVICES=0

CHECKPOINTS_DIR=$LIT_CHECKPOINTS
HF_TOKEN=$(cat hf_token.txt)
model="tinyllama"

# Supported models
declare -A model2checkpoint=(
    ["tinyllama"]="$CHECKPOINTS_DIR/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    ["phi3"]="$CHECKPOINTS_DIR/microsoft/Phi-3-mini-4k-instruct"
    ["llama3"]="$CHECKPOINTS_DIR/meta-llama/Meta-Llama-3-8B"
)
if [ ! -d ${model2checkpoint[$model]} ]; then
    litgpt download ${model2checkpoint[$model]} --checkpoint_dir $CHECKPOINTS_DIR --access_token $HF_TOKEN
fi

# Datasets
declare -a DATASETS=(sst2 20newsgroups agnews )

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
    ["sst2_8"]="639 923"
    ["sst2_32"]="1564 1738"
    ["sst2_512"]="111 121"
    ["agnews_4"]="295 926"
    ["agnews_16"]="738 564"
    ["agnews_256"]="493 821"
    ["dbpedia_2"]="435 927"
    ["dbpedia_8"]="338 364"
    ["dbpedia_128"]="129 131"
    ["20newsgroups_2"]="435 927"
    ["20newsgroups_8"]="338 364"
    ["20newsgroups_128"]="129 131"
    ["banking77_1"]="322 444"
    ["banking77_4"]="295 926"
    ["banking77_64"]="131 888"
)

source ./scripts/prepare_datasets.sh
source ./scripts/prepare_prompts.sh
for dataset in ${DATASETS[@]}; do
    source ./scripts/generative_no_adaptation.sh
    source ./scripts/affine_calibration.sh
    source ./scripts/lora.sh
    source ./scripts/lora_no_es.sh
    source ./scripts/lora_norm.sh
    source ./scripts/lora_plus_affine_cal_no_es.sh
    source ./scripts/lora_norm_plus_affine_cal_no_es.sh
done