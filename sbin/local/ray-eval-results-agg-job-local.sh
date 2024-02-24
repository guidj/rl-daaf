#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..

DATA_DIR=$1
INPUT_DIR=$HOME/fs/daaf/exp/evaljob/logs/${DATA_DIR}
OUTPUT_DIR=$HOME/fs/daaf/exp/evaljob/${DATA_DIR}/`date +%s`
rm -rf ${OUTPUT_DIR}
python $PARENT_DIR/src/daaf/policyeval/results_agg_pipeline.py \
    --input-dir=$INPUT_DIR \
    --output-dir=$OUTPUT_DIR
