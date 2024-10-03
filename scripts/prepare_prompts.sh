#! /bin/bash -ex

DATA_DIR=outputs/data
OUTPUT_DIR=outputs/prompts/generative
PROMPTS_TEMPLATES_DIR=prompts
max_characters=500

for dataset in ${DATASETS[@]}; do
    mkdir -p $OUTPUT_DIR/$dataset
    if [ ! -f $OUTPUT_DIR/$dataset/train_prompt.jsonl ] && [ ! -f $OUTPUT_DIR/$dataset/test_prompt.jsonl ]; then
        python -m llmcal2.scripts.prepare_prompt \
            --data_dir $DATA_DIR/$dataset \
            --prompt_template $PROMPTS_TEMPLATES_DIR/basic_$dataset.yaml \
            --output_dir $OUTPUT_DIR/$dataset \
            --max_characters $max_characters
    fi
done