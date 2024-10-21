#! /bin/bash -ex

OUTPUT_DIR=outputs/data
LISTS_DIR=lists

for dataset in ${DATASETS[@]}; do
    test_list=${dataset2testlist[$dataset]}
    mkdir -p $OUTPUT_DIR/$dataset
    if [ ! -f $OUTPUT_DIR/$dataset/train--all.csv ] && [ ! -f $OUTPUT_DIR/$dataset/${test_list}.csv ]; then
        python -m llmcal2.scripts.prepare_dataset --dataset $dataset --lists_dir $LISTS_DIR/$dataset --output_dir $OUTPUT_DIR/$dataset
    fi
done