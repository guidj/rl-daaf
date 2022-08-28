#!/bin/bash
set -eo

DIR=$(dirname $0)
cd ${DIR}/../../

module="onpolicy_eval"
problem=ABCSeq
episodes=1000
reward_period=2
script="./src/daaf/periodic_reward/${module}.py"
output="./data/$problem/L${level}_P${reward_period}_EPS${episodes}"
args="length=4"

for cu_step_mapper in identity-mapper reward-imputation-mapper reward-estimation-ls-mapper; do
    python $script --problem $problem \
        --output=$output/$cu_step_mapper/ \
        --num-episodes=$episodes \
        --cu-step-mapper=$cu_step_mapper \
        --reward-period=$reward_period \
        --mdp-stats-path=$mdp_stats_path $args
done
