#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..

TIMESTAMP=`date +%s`
python $PARENT_DIR/src/daaf/rewardest/estjob.py \
--num-runs=10 \
--max-episodes=2500 \
--output-dir=$HOME/fs/daaf/exp/reward-estjob/logs/$TIMESTAMP \
--log-episode-frequency=10
