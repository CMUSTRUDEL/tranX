#!/bin/bash

model_name=$(basename $1)

python exp.py \
    --cuda \
    --mode test \
    --load-model $1 \
    --beam-size 5 \
    --parser wikisql_parser \
    --evaluator wikisql_evaluator \
    --sql-db-file data/wikisql/test.db \
    --test-file data/wikisql/test.bin \
    --save-decode-to decodes/wikisql/${model_name}.decode \
    --decode-max-time-step 50
