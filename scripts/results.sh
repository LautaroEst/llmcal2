#! /bin/bash -ex

mkdir -p outputs/results
python -m llmcal2.scripts.plot_methods_vs_size \
    --model tinyllama \
    --metrics ner,nce,cal_loss_nce \
    --datasets "sst2,agnews,dbpedia,20newsgroups,banking77" \
    --methods "lora,lora_norm,dp_calibration,temp_scaling,no_adaptation,lora_plus_dp_calibration_no_es,lora_norm_plus_dp_calibration_no_es" \
    --results_dir outputs/adaptation \
    --output_dir outputs/results \
    --overwrite
