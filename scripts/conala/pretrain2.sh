#!/bin/bash
set -e

echo "****Pretraining with API docs first****"
seed=0
mined_num=$1
ret_method=$2
vocab="data/conala/vocab.src_freq3.code_freq3.mined_${mined_num}.${ret_method}5.bin"
train_file="data/conala/${ret_method}5.bin"
dev_file="data/conala/dev.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
lr=0.001
lr_decay=0.5
batch_size=32
max_epoch=80
beam_size=15
lstm='lstm'  # lstm
lr_decay_after_epoch=15
model_name=api.${ret_method}.dr${dropout}.lr${lr}.lr_de${lr_decay}.lr_da${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).seed${seed}

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

echo "****Pretraining with mined data second****"

pretrained_model_name=${model_name}
vocab="data/conala/vocab.src_freq3.code_freq3.mined_${mined_num}.${ret_method}5.bin"
finetune_file="data/conala/mined_${mined_num}.bin"
dev_file="data/conala/dev.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
lr=0.001
lr_decay=0.5
beam_size=15
batch_size=64
max_epoch=80
lstm='lstm'  # lstm
lr_decay_after_epoch=15
model_name=api.${ret_method}.mined.dr${dropout}.lr${lr}.lr_de${lr_decay}.lr_da${lr_decay_after_epoch}.beam${beam_size}.seed${seed}.mined_${mined_num}

echo "**** Writing results to logs/conala/${model_name}.log ****"
mkdir -p logs/conala
echo commit hash: "$(git rev-parse HEAD)" > logs/conala/"${model_name}".log

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch-size ${batch_size} \
    --evaluator conala_evaluator \
    --asdl-file asdl/lang/py3/py3_asdl.simplified.txt \
    --transition-system python3 \
    --train-file ${finetune_file} \
    --dev-file ${dev_file} \
    --pretrain saved_models/conala/${pretrained_model_name}.bin \
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
