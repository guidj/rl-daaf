# Ray

The experiments and results aggregation pipelines are executed using ray.
One can rely on the local machine or use a remote cluster.

## Spinning up clusters

This depends on the ray setup you have.
The important thing to take note of is the version of ray running on the cluster.
Ray expects `ray job submit` commands to run using the same version of ray
that's on the cluster.

## Submitting jobs

The simplest way is to use the `ray job submit` command.
Examples of it for a local run can be found in [sbin](../sbin/).

In case the cluster is using a different version of ray, it may be
simplest to create a separate environment for local jobs
and one for remote jobs, each with the suitable version of ray.


Note: when ray submits a ray, it respects the local .gitignore.
So any files required shouldn't match patterns in .gitignore.


Example remote submission to a cluster:

```
CLUSTER_URI=http://ip:port
RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1

TIMESTAMP=`date +%s`
ray job submit \
    --address=$WORKING_DIR \
    --working-dir=$WORKING_DIR \
    --runtime-env-json='{"py_modules":["src/daaf"], "pip": "ray-env-requirements.txt"}' \
    -- \
    python $WORKING_DIR/src/daaf/policyeval/evaljob.py \
    --config-path=$WORKING_DIR/experiments/policyeval/experiments-sample.csv \
    --envs-path=$WORKING_DIR/experiments/policyeval/envs-sample.json \
    --num-runs=3 \
    --num-episodes=7500 \
    --assets-dir=$GCS_PREFIX/assets \
    --output-dir=$GCS_PREFIX/exp/logs/$TIMESTAMP \
    --task-prefix $TIMESTAMP \
    --log-episode-frequency=10

```