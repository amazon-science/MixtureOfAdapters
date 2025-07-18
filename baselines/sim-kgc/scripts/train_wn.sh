#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

mkdir -p ${OUTPUT_DIR}

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task ${TASK} \
--batch-size 256 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--pre-batch 0 \
--finetune-t \
--epochs 50 \
--workers 4 \
--encoding_type tri_encoder \
--triencoder_head moa \
--hypernet_scaler 1 \
--freeze_encoder \
--transform \
--max-to-keep 3 "$@" &> "${OUTPUT_DIR}/log.txt"
