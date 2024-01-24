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
