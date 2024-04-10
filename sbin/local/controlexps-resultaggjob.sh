#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..

DATA_DIR=$1
INPUT_DIR=$HOME/fs/daaf/exp/controljob/logs/${DATA_DIR}
OUTPUT_DIR=$HOME/fs/daaf/exp/controljob/agg/${DATA_DIR}/`date +%s`
rm -rf ${OUTPUT_DIR}
python $PARENT_DIR/src/daaf/controlexps/results_agg_pipeline.py \
    --input-dir=$INPUT_DIR \
    --output-dir=$OUTPUT_DIR
