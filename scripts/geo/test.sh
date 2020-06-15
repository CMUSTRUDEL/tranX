#!/bin/bash

model_name=$(basename $1)

python exp.py \
    --cuda \
    --mode test \
    --load-model $1 \
    --beam-size 5 \
    --test-file data/geo/test.bin \
    --decode-max-time-step 110
