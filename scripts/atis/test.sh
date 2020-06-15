#!/bin/bash

model_name=$(basename $1)

python exp.py \
    --mode test \
    --load-model $1 \
    --beam-size 5 \
    --test-file data/atis/test.bin \
    --save-decode-to decodes/atis/${model_name}.test.decode \
    --decode-max-time-step 110
