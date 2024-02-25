#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..

TIMESTAMP=`date +%s`
python $PARENT_DIR/src/daaf/baselines.py \
    --assets-dir=$HOME/fs/daaf/exp/dynamicprog/$TIMESTAMP