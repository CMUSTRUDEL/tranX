#!/bin/bash
set -e

seed=${1:-0}
vocab="vocab.freq2.bin"
train_file="train.bin"
dev_file="dev.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=32
type_embed_size=32
lr_decay=0.5
beam_size=5
lstm='lstm'
ls=0.1
model_name=model.atis.sup.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dropout${dropout}.lr_decay${lr_decay}.beam${beam_size}.${vocab}.${train_file}.glorot.with_par_info.no_copy.ls${ls}.seed${seed}

echo "**** Writing results to logs/atis/${model_name}.log ****"
mkdir -p logs/atis
echo commit hash: `git rev-parse HEAD` > logs/atis/${model_name}.log

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch-size 10 \
    --asdl-file asdl/lang/lambda_dcs/lambda_asdl.txt \
    --transition-system lambda_dcs \
    --train-file data/atis/${train_file} \
    --dev-file data/atis/${dev_file} \
    --vocab data/atis/${vocab} \
    --lstm ${lstm} \
    --primitive-token-label-smoothing ${ls} \
    --no-parent-field-type-embed \
    --no-parent-production-embed \
    --hidden-size ${hidden_size} \
    --att-vec-size ${hidden_size} \
    --embed-size ${embed_size} \
    --action-embed-size ${action_embed_size} \
    --field-embed-size ${field_embed_size} \
    --type-embed-size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max-num-trial 5 \
    --glorot-init \
    --no-copy \
    --lr-decay ${lr_decay} \
    --beam-size ${beam_size} \
    --decode-max-time-step 110 \
    --log-every 50 \
    --save-to saved_models/atis/${model_name} 2>&1 | tee -a logs/atis/${model_name}.log

. scripts/atis/test.sh saved_models/atis/${model_name}.bin 2>&1 | tee -a logs/atis/${model_name}.log
