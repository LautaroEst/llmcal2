#! /bin/bash -ex

OUTPUT_DIR=outputs/data

for dataset in $DATASETS; do
    mkdir -p $OUTPUT_DIR/$dataset
    # check whether $OUTPUT_DIR/$dataset/train.csv and $OUTPUT_DIR/$dataset/test.csv exist:
    if [ ! -f $OUTPUT_DIR/$dataset/train.csv ] && [! -f $OUTPUT_DIR/$dataset/test.csv ]; then
        python -m llmcal2.scripts.prepare_dataset --dataset $dataset --lists_dir lists/$dataset --output_dir $OUTPUT_DIR/$dataset
    fi
done