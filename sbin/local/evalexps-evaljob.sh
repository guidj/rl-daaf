#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..

TIMESTAMP=`date +%s`
python $PARENT_DIR/src/daaf/evalexps/evaljob.py \
    --config-path=$PARENT_DIR/experiments/policyeval/experiments-sample.csv \
    --envs-path=$PARENT_DIR/experiments/policyeval/envs-sample.json \
    --num-runs=1 \
    --num-episodes=10 \
    --assets-dir=$HOME/fs/daaf/assets \
    --output-dir=$HOME/fs/daaf/exp/evaljob/logs/$TIMESTAMP \
    --task-prefix $TIMESTAMP \
    --log-episode-frequency=10 \
    --metrics-last-k-episodes=100