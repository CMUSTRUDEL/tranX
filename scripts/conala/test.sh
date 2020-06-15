#!/bin/bash

test_file="data/conala/test.var_str_sep.bin"

python exp.py \
    --mode test \
    --load-model $1 \
    --beam-size 15 \
    --test-file ${test_file} \
    --evaluator conala_evaluator \
    --save-decode-to decodes/conala/$(basename $1).test.decode \
    --decode-max-time-step 100

