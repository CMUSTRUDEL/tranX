#!/usr/bin/env bash

# Usage: ./scripts/c/train_no_treebpe.sh var_name beam_size
#   - var_name: decompiled or original

set -e

seed=19260817
vocab="tranx_data/vocab.pkl"
train_file="tranx_data/"
dev_file="tranx_data/dev/"
batch_size=12
max_tokens_per_batch=4096
dropout=0.3
hidden_size=512
poswise_ff_dim=2048
embed_size=512
action_embed_size=256
field_embed_size=128
type_embed_size=128
decode_max_time_step=1000
max_src_len=512
max_tgt_actions=512
valid_every_iters=20000
encoder_layers=6
lr=0.001
lr_decay=0.5
beam_size=$2
n_procs=4
var_name=$1
tree_bpe_model=none
src_repr_mode="action_seq"
model_name=model.transformer.no_treebpe.beam_size${beam_size}.var_${var_name}.input_${src_repr_mode}.$(basename ${vocab}).$(basename ${train_file})

shift 2

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
    --src-repr-mode ${src_repr_mode} \
    --encoder 'transformer' \
    --encoder-layers ${encoder_layers} \
    --poswise-ff-dim ${poswise_ff_dim} \
    --decoder 'transformer' \
    --no-parent-field-type-embed \
    --no-parent-production-embed \
    --max-src-len ${max_src_len} \
    --max-actions ${max_tgt_actions} \
    --max-tokens-per-batch ${max_tokens_per_batch} \
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
    --allow-incomplete-hypotheses \
    --save-to "saved_models/c/${model_name}" \
    --save-all-models \
    --write-log-to "logs/c/${model_name}.log" \
    "$@"
