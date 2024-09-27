
OUTPUTS_DIR=$LIT_CHECKPOINTS
HF_TOKEN=$(cat hf_token.txt)

# MODELS="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T microsoft/Phi-3-mini-4k-instruct meta-llama/Meta-Llama-3-8B"
MODELS="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T "
for model in $MODELS; do
    litgpt download $model --checkpoint_dir $OUTPUTS_DIR --access_token $HF_TOKEN
done