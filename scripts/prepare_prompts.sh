#! /bin/bash -ex

OUTPUT_DIR=outputs/prompts
DATASETS="sst2 agnews dbpedia 20newsgroups banking77"

for dataset in $DATASETS; do
    mkdir -p $OUTPUT_DIR/generative/$dataset
    python -m llmcal2.scripts.prepare_prompt \
        --data_dir outputs/data/$dataset \
        --prompt_template prompts/basic_$dataset.yaml \
        --output_dir $OUTPUT_DIR/generative/$dataset
done