#!/usr/bin/env bash
set -e

seed=${1:-19260817}
vocab="tranx_data/vocab.pkl"
train_file="tranx_data/"
dev_file="data/django/dev.bin"
test_file="data/django/test.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
ptrnet_hidden_dim=32
decode_max_time_step=500
lr=0.001
lr_decay=0.5
beam_size=15
lstm='lstm'  # lstm
model_name=model.sup.c.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dropout${dropout}.lr${lr}.lr_decay${lr_decay}.beam_size${beam_size}.$(basename ${vocab}).$(basename ${train_file}).glorot.par_state_w_field_embe.seed${seed}

echo "**** Writing results to logs/c/${model_name}.log ****"
mkdir -p logs/c
echo commit hash: `git rev-parse HEAD` > logs/c/${model_name}.log

python exp.py \
    --seed ${seed} \
    --mode train \
    --batch_size 10 \
    --asdl_file asdl/lang/c/c_asdl.txt \
    --dataset c_dataset \
    --transition_system c \
    --evaluator c_evaluator \
    --train_file ${train_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --glorot_init \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --beam_size ${beam_size} \
    --log_every 50 \
    --num_workers 0 \
    --decode_max_time_step ${decode_max_time_step} \
    --save_to "saved_models/c/${model_name}" 2>&1 | tee -a "logs/c/${model_name}.log"
