#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..

TIMESTAMP=`date +%s`
python $PARENT_DIR/src/daaf/policyeval/evaljob.py \
    --config-path=$PARENT_DIR/experiments/daaf/experiments-sample.csv \
    --envs-path=$PARENT_DIR/experiments/daaf/envs-sample.json \
    --num-runs=3 \
    --num-episodes=10 \
    --assets-dir=$HOME/fs/daaf/assets \
    --output-dir=$HOME/fs/daaf/exp/logs/$TIMESTAMP \
    --task-prefix $TIMESTAMP \
    --log-episode-frequency=1 \
    --num-tasks=4
