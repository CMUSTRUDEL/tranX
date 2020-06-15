#!/bin/bash

test_file="data/django/test.bin"

python exp.py \
	--cuda \
    --mode test \
    --load-model $1 \
    --beam-size 15 \
    --test-file ${test_file} \
    --save-decode-to decodes/django/$(basename $1).test.decode \
    --decode-max-time-step 100
