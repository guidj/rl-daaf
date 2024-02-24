#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..

TIMESTAMP=`date +%s`
python $PARENT_DIR/src/daaf/policyeval/evaljob.py \
    --config-path=$PARENT_DIR/experiments/policyeval/experiments-sample.csv \
    --envs-path=$PARENT_DIR/experiments/policyeval/envs-sample.json \
    --num-runs=3 \
    --num-episodes=7500 \
    --assets-dir=$HOME/fs/daaf/assets \
    --output-dir=$HOME/fs/daaf/exp/evaljob/logs/$TIMESTAMP \
    --task-prefix $TIMESTAMP \
    --log-episode-frequency=10
