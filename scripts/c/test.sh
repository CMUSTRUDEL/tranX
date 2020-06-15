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
    --test_file ${test_file} \
    --load_model "${load_model}" \
    --asdl_file asdl/lang/c/c_asdl.txt \
    --dataset c_dataset \
    --parser c_parser \
    --transition_system c \
    --evaluator c_evaluator \
    --vocab ${vocab} \
    --beam_size ${beam_size} \
    --num_workers ${n_procs} \
    --variable_name ${var_name} \
    --tree_bpe_model ${tree_bpe_model} \
    --decode_max_time_step ${decode_max_time_step} \
    --allow_incomplete_hypotheses \
    --save_decode_to decodes/c/$(basename "${load_model}").test.beam_size${beam_size}.max_time${decode_max_time_step}.decode.pkl \
    --save_decode_text_to decodes/c/$(basename "${load_model}").test.beam_size${beam_size}.max_time${decode_max_time_step}.decode.txt
