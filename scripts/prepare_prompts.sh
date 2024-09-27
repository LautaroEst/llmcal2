#! /bin/bash -ex

OUTPUT_DIR=outputs/prompts

for dataset in $DATASETS; do
    mkdir -p $OUTPUT_DIR/generative/$dataset
    if [ ! -f $OUTPUT_DIR/generative/$dataset/train_prompt.jsonl ] && [ ! -f $OUTPUT_DIR/generative/$dataset/test_prompt.jsonl ]; then
        python -m llmcal2.scripts.prepare_prompt \
            --data_dir outputs/data/$dataset \
            --prompt_template prompts/basic_$dataset.yaml \
            --output_dir $OUTPUT_DIR/generative/$dataset
    fi
    
done