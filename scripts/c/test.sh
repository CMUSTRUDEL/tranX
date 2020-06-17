#!/usr/bin/env bash
set -e

load_model=$1
seed=19260817
vocab="tranx_data/vocab.pkl"
test_file="tranx_data/test/"
n_procs=1
var_name="original"
tree_bpe_model="tranx_data/tree_bpe_model.pkl"
decode_max_time_step=1000
beam_size=1

python exp.py \
    --verbose \
    --seed ${seed} \
    --mode test \
    --test-file ${test_file} \
    --load-model "${load_model}" \
    --asdl-file asdl/lang/c/c_asdl.txt \
    --dataset c_dataset \
    --parser c_parser \
    --transition-system c \
    --evaluator c_evaluator \
    --vocab ${vocab} \
    --beam-size ${beam_size} \
    --num-workers ${n_procs} \
    --variable-name ${var_name} \
    --tree-bpe-model ${tree_bpe_model} \
    --decode-max-time-step ${decode_max_time_step} \
    --allow-incomplete-hypotheses \
    --save-decode-to decodes/c/$(basename "${load_model}").test.beam_size${beam_size}.max_time${decode_max_time_step}.decode.pkl \
    --save-decode-text-to decodes/c/$(basename "${load_model}").test.beam_size${beam_size}.max_time${decode_max_time_step}.decode.txt
