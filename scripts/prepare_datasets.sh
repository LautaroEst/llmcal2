#! /bin/bash -ex

OUTPUT_DIR=outputs/data
DATASETS="sst2 agnews dbpedia 20newsgroups banking77"

for dataset in $DATASETS; do
    mkdir -p $OUTPUT_DIR/$dataset
    python -m llmcal2.scripts.prepare_dataset --dataset $dataset --lists_dir lists/$dataset --output_dir $OUTPUT_DIR/$dataset
done