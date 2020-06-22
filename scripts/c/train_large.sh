#!/usr/bin/env bash
set -e

seed=19260817
vocab="tranx_data/vocab.pkl"
train_file="tranx_data/"
dev_file="tranx_data/dev/"
batch_size=12
dropout=0.3
hidden_size=512
embed_size=256
action_embed_size=256
field_embed_size=128
type_embed_size=128
decode_max_time_step=500
valid_every_iters=20000
lr=0.001
lr_decay=0.5
beam_size=1
n_procs=3
var_name="original"
tree_bpe_model="tranx_data/tree_bpe_model.pkl"
model_name=model.sup.c.var_${var_name}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dropout${dropout}.lr${lr}.lr_decay${lr_decay}.beam_size${beam_size}.$(basename ${vocab}).$(basename ${train_file}).seed${seed}

echo "**** Writing results to logs/c/${model_name}.log ****"
mkdir -p logs/c
echo commit hash: `git rev-parse HEAD` > logs/c/${model_name}.log

python exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch-size ${batch_size} \
    --asdl-file asdl/lang/c/c_asdl.txt \
    --dataset c_dataset \
    --parser c_parser \
    --transition-system c \
    --evaluator c_evaluator \
    --train-file ${train_file} \
    --dev-file ${dev_file} \
    --vocab ${vocab} \
    --lstm 'lstm' \
    --no-parent-field-type-embed \
    --no-parent-production-embed \
    --hidden-size ${hidden_size} \
    --embed-size ${embed_size} \
    --action-embed-size ${action_embed_size} \
    --field-embed-size ${field_embed_size} \
    --type-embed-size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max-num-trial 5 \
    --glorot-init \
    --lr ${lr} \
    --lr-decay ${lr_decay} \
    --beam-size ${beam_size} \
    --valid-every-epoch -1 \
    --valid-every-iters ${valid_every_iters} \
    --log-every 10 \
    --num-workers ${n_procs} \
    --variable-name ${var_name} \
    --tree-bpe-model ${tree_bpe_model} \
    --decode-max-time-step ${decode_max_time_step} \
    --save-to "saved_models/c/${model_name}" \
    --write-log-to "logs/c/${model_name}.log" \
    "$@"
