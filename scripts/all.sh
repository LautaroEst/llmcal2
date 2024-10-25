#!/bin/bash -ex

./scripts/prepare_data.sh
./scripts/no_adaptation.sh
./scripts/lora.sh
./scripts/lora_plus_dp_cal.sh