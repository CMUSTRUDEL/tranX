#!/usr/bin/env bash
set -e

load_model=$1
seed=19260817
vocab="tranx_data/vocab.pkl"
test_file="tranx_data/test/"
decode_max_time_step=500
beam_size=1

python exp.py \
    --verbose \
    --seed ${seed} \
    --mode test \
    --test_file ${test_file} \
    --load_model "${load_model}" \
    --asdl_file asdl/lang/c/c_asdl.txt \
    --dataset c_dataset \
    --parser c_parser \
    --transition_system c \
    --evaluator c_evaluator \
    --vocab ${vocab} \
    --beam_size ${beam_size} \
    --num_workers 0 \
    --decode_max_time_step ${decode_max_time_step} \
    --save_decode_to decodes/c/$(basename "${load_model}").test.decode
