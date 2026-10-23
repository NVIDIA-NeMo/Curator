#!/bin/bash

source local_env.sh
# Table 1
SIMULATED_OUT="${SIM_OUT_ROOT}/nsf"
CONFIGS=("flat" "callhome" "callhome_boosted" "ami_estimate" "nsf_estimate" "nsf_estimate_markov" "nsf_estimate_boosted")

for CONFIG in "${CONFIGS[@]}"; do
    echo "Processing config: $CONFIG"
    python "${FastMSS_ROOT}/sim.py" --config-path=config/table1 --config-name "$CONFIG"
    python "${MADP_ROOT}/src/pre_segment_using_alignments.py" \
        --input "$SIMULATED_OUT/${CONFIG}/manifests/synth-notsofar1_ihm-train-cuts.jsonl.gz" \
        --output "$SIMULATED_OUT/${CONFIG}/manifests/synth-notsofar1_ihm-train-cuts_30s.jsonl.gz" \
        --max_len 30
done

# Table 2
CONFIGS=(
    "librispeech librispeech"
    "voxpopuli voxpopuli"
    "oto_speech oto"
    "ami ami-ihm"
) #NSF prepared from previous step

for tuple in "${CONFIGS[@]}"; do
    # Unpack the tuple into two separate variables
    read -r CONFIG CUTSET <<< "$tuple"

    echo "Processing config: $CONFIG"

    python "${FastMSS_ROOT}/sim.py" --config-path=config/table2/clean --config-name "$CONFIG"

    python "${MADP_ROOT}/src/pre_segment_using_alignments.py" \
        --input "$SIMULATED_OUT/${CONFIG}/manifests/synth-${CUTSET}-train-cuts.jsonl.gz" \
        --output "$SIMULATED_OUT/${CONFIG}/manifests/synth-${CUTSET}-train-cuts_30s.jsonl.gz" \
        --max_len 30
done
