#!/bin/bash -ex

./scripts/prepare_data.sh
./scripts/no_adaptation.sh
./scripts/dp_calibration.sh
./scripts/few_shot.sh
./scripts/few_shot_plus_dp_cal.sh
./scripts/lora_matched.sh
./scripts/lora_matched_plus_dp_cal.sh
./scripts/lora_matched_no_es.sh
./scripts/lora_mismatched.sh
./scripts/lora_mismatched_plus_dp_cal.sh
./scripts/lora_mismatched_few_shot.sh
./scripts/lora_mismatched_few_shot_plus_dp_cal.sh
./scripts/lora_all.sh
./scripts/lora_all_plus_dp_cal.sh
./scripts/bert.sh
