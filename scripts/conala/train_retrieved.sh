#!/bin/bash
set -e

seed=0
mined_num=$1
ret_method=$2
freq=3
vocab="data/conala/vocab.src_freq${freq}.code_freq${freq}.mined_${mined_num}.${ret_method}5.bin"
train_file="data/conala/pre_${mined_num}_${ret_method}5.bin"
dev_file="data/conala/dev.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
lr=0.001
lr_decay=0.5
batch_size=64
max_epoch=80
beam_size=15
lstm='lstm'  # lstm
lr_decay_after_epoch=15
model_name=ret.dr${dropout}.lr${lr}.lr_de${lr_decay}.lr_da${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).seed${seed}

echo "**** Writing results to logs/conala/${model_name}.log ****"
mkdir -p logs/conala
echo commit hash: `git rev-parse HEAD` > logs/conala/${model_name}.log

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch-size ${batch_size} \
    --evaluator conala_evaluator \
    --asdl-file asdl/lang/py3/py3_asdl.simplified.txt \
    --transition-system python3 \
    --train-file ${train_file} \
    --dev-file ${dev_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
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
    --lr-decay-after-epoch ${lr_decay_after_epoch} \
    --max-epoch ${max_epoch} \
    --beam-size ${beam_size} \
    --log-every 50 \
    --save-to saved_models/conala/${model_name} 2>&1 | tee logs/conala/${model_name}.log

. scripts/conala/test.sh saved_models/conala/${model_name}.bin 2>&1 | tee -a logs/conala/${model_name}.log
