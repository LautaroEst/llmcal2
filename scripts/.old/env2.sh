CHECKPOINTS_DIR=outputs/checkpoints
HF_TOKEN=$(cat hf_token.txt)
model="llama3.2-1b"
# model="pythia-14m"
# model=tinyllama

# Reproducibility
base_seed=2834
num_seeds=5

# Supported models
declare -A model2checkpoint=(
    ["pythia-14m"]="EleutherAI/pythia-14m"
    ["llama3.2-1b"]="meta-llama/Llama-3.2-1B"
    ["llama3.2-1b-instruct"]="meta-llama/Llama-3.2-1B-Instruct"
    ["tinyllama"]="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)
mkdir -p $CHECKPOINTS_DIR
if [ ! -d $CHECKPOINTS_DIR/${model2checkpoint[$model]} ]; then
    litgpt download ${model2checkpoint[$model]} --checkpoint_dir $CHECKPOINTS_DIR --access_token $HF_TOKEN
    rm -rf $CHECKPOINTS_DIR/${model2checkpoint[$model]}/*.bin
fi
if [ ! -z ${model2checkpoint[${model}-instruct]} ]; then
    if [ ! -d $CHECKPOINTS_DIR/${model2checkpoint[${model}-instruct]} ]; then
        litgpt download ${model2checkpoint[${model}-instruct]} --checkpoint_dir $CHECKPOINTS_DIR --access_token $HF_TOKEN
        rm -rf $CHECKPOINTS_DIR/${model2checkpoint[${model}-instruct]}/*.bin
    fi
fi

# Datasets
declare -a DATASETS=(sst2 agnews dbpedia 20newsgroups banking77)

declare -A dataset2trainsize=(
    ["sst2"]=16
    ["agnews"]=16
    ["dbpedia"]=28
    ["20newsgroups"]=40
    ["banking77"]=77
)
# declare -A dataset2trainsize=(
#     ["sst2"]=64
#     ["agnews"]=64
#     ["dbpedia"]=112
#     ["20newsgroups"]=160
#     ["banking77"]=308
# )
# declare -A dataset2trainsize=(
#     ["sst2"]=1024
#     ["agnews"]=1024
#     ["dbpedia"]=1792
#     ["20newsgroups"]=2560
#     ["banking77"]=4928
# )

declare -A dataset2testsize=(
    ["sst2"]=400
    ["agnews"]=400
    ["dbpedia"]=700
    ["20newsgroups"]=800
    ["banking77"]=1000
)

max_seq_length=2048
inference_max_seq_len=20000

# losses_options=(ans fs norm-5 norm-15)
losses_options=(ans fs)

declare -a N_SHOTS=(4 16)

export CUDA_VISIBLE_DEVICES=0