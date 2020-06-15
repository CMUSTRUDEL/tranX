#!/bin/bash

decode_file=$1
model_file=$2

python exp.py \
    --cuda \
    --mode test \
    --load-model "${model_file}" \
    --beam-size 15 \
    --test-file "${decode_file}" \
    --evaluator conala_evaluator \
    --save-decode-to "decodes/conala/$(basename ${model_file}).$(basename ${decode_file}).decode" \
    --decode-max-time-step 100
