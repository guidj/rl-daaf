# RL with Delayed, Aggregated and Anonymous Feedback (DAAF)

Code for experiments on RL policy evaluation with delayed, aggregated and anonymous feedback.


Contains
  - Algorithms implementation (ZI-M, LEAST...)
  - Notebooks with analysis results

Missing
  - Pipeline that aggregates results from multiple results (output is used in notebooks)


## Dev Env
First, make sure the following python development tools are installed:
  - pip
  - pip-tools(==7.3.0)

Then, in a virtual environment, run pip-compile and install:

```
$ make pip-compile
$ make pip-install
```

These should install all the requirements dependencies for development.

## Dependencies

The dependecy files map to a purpose as follows:

  - [requirements.in](requirements.in): packages for the experiments.
  - [test-requirements.in](test-requirements.in): for running tests.
  - [nb-requirements.in](nb-requirements.in): for jupyter notebooks.
  - [rendering-requirements.in](rendering-requirements.in): for environments can be rendered in a graphical interface, with OpenGL.
  - [ray-env-requirements.in](ray-env-requirements.in): for ray in a cluster environment. During compilation with `pip-compile`, it's best to exclude the version of ray (see [Makefile](Makefile)).

All requirements files are compiled using `pip-compile`.