#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..

TIMESTAMP=`date +%s`
python $PARENT_DIR/src/daaf/controlexps/controljob.py \
    --config-path=$PARENT_DIR/experiments/policycontrol/experiments-sample.csv \
    --envs-path=$PARENT_DIR/experiments/policycontrol/envs-sample.json \
    --num-runs=3 \
    --num-episodes=10 \
    --assets-dir=$HOME/fs/daaf/assets \
    --output-dir=$HOME/fs/daaf/exp/controljob/logs/$TIMESTAMP \
    --task-prefix $TIMESTAMP \
    --log-episode-frequency=1
