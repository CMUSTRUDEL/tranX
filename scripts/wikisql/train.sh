#!/bin/bash
set -e

seed=${1:-0}
vocab="vocab.bin"
train_file="train.bin"
dropout=0.3
hidden_size=256
embed_size=100
action_embed_size=100
field_embed_size=32
type_embed_size=32
lr_decay=0.5
beam_size=5
patience=5
lstm='lstm'
col_att='affine'
model_name=model.wikisql.sup.exe_acc.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dropout${dropout}.lr_decay${lr_decay}.pat${patience}.beam${beam_size}.${vocab}.${train_file}.col_att_${col_att}.glorot.no_par_info.seed${seed}

echo "**** Writing results to logs/wikisql/${model_name}.log ****"
mkdir -p logs/wikisql
echo commit hash: `git rev-parse HEAD` > logs/wikisql/${model_name}.log

echo `which python`

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch-size 64 \
    --parser wikisql_parser \
    --asdl-file asdl/lang/sql/sql_asdl.txt \
    --transition-system sql \
    --evaluator wikisql_evaluator \
    --train-file data/wikisql/${train_file} \
    --dev-file data/wikisql/dev.bin \
    --sql-db-file data/wikisql/dev.db \
    --vocab data/wikisql/${vocab} \
    --glove-embed-path data/contrib/glove.6B.100d.txt \
    --lstm ${lstm} \
    --column-att ${col_att} \
    --no-parent-state \
    --no-parent-field-embed \
    --no-parent-field-type-embed \
    --no-parent-production-embed \
    --hidden-size ${hidden_size} \
    --embed-size ${embed_size} \
    --action-embed-size ${action_embed_size} \
    --field-embed-size ${field_embed_size} \
    --type-embed-size ${type_embed_size} \
    --dropout ${dropout} \
    --patience ${patience} \
    --max-num-trial 5 \
    --lr-decay ${lr_decay} \
    --glorot-init \
    --beam-size ${beam_size} \
    --eval-top-pred-only \
    --decode-max-time-step 50 \
    --log-every 10 \
    --save-to saved_models/wikisql/${model_name} 2>&1 | tee -a logs/wikisql/${model_name}.log

. scripts/wikisql/test.sh saved_models/wikisql/${model_name}.bin 2>&1 | tee -a logs/wikisql/${model_name}.log
