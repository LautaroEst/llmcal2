#! /bin/bash -ex

mkdir -p outputs/results
python -m llmcal2.scripts.plot_methods_vs_size \
    --model tinyllama \
    --metrics ner,nce,cal_loss_nce \
    --datasets "sst2," \
    --methods "lora,dp_calibration,temp_scaling,no_adaptation" \
    --results_dir outputs/adaptation \
    --output_dir outputs/results \
    --overwrite
