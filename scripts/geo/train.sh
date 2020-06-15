#!/bin/bash
set -e

seed=${1:-0}
vocab="data/geo/vocab.freq2.bin"
train_file="data/geo/train.bin"
dropout=0.5
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=32
type_embed_size=32
lr_decay=0.985
lr_decay_after_epoch=20
max_epoch=200
patience=1000   # disable patience since we don't have dev set
beam_size=5
batch_size=10
lr=0.0025
ls=0.1
lstm='lstm'
model_name=model.geo.sup.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.field${field_embed_size}.type${type_embed_size}.drop${dropout}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glorot.no_par_info.no_copy.ls${ls}.seed${seed}

echo "**** Writing results to logs/geo/${model_name}.log ****"
mkdir -p logs/geo
echo commit hash: `git rev-parse HEAD` > logs/geo/${model_name}.log

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch-size ${batch_size} \
    --asdl-file asdl/lang/lambda_dcs/lambda_asdl.txt \
    --transition-system lambda_dcs \
    --train-file ${train_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --primitive-token-label-smoothing ${ls} \
    --no-parent-field-type-embed \
    --no-parent-production-embed \
    --no-parent-field-embed \
    --no-parent-state \
    --hidden-size ${hidden_size} \
    --embed-size ${embed_size} \
    --action-embed-size ${action_embed_size} \
    --field-embed-size ${field_embed_size} \
    --type-embed-size ${type_embed_size} \
    --dropout ${dropout} \
    --patience ${patience} \
    --max-epoch ${max_epoch} \
    --lr ${lr} \
    --no-copy \
    --lr-decay ${lr_decay} \
    --lr-decay-after-epoch ${lr_decay_after_epoch} \
    --decay-lr-every-epoch \
    --glorot-init \
    --beam-size ${beam_size} \
    --decode-max-time-step 110 \
    --log-every 50 \
    --save-to saved_models/geo/${model_name} 2>&1 | tee -a logs/geo/${model_name}.log

. scripts/geo/test.sh saved_models/geo/${model_name}.bin 2>&1 | tee -a logs/geo/${model_name}.log
