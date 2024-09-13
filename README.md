# RL with Delayed, Aggregated and Anonymous Feedback (DAAF)

Code for experiments on policy control and evaluation in Reinforcement Learning with delayed, aggregated and anonymous feedback.


## Delayed, aggregate, anoynmous feedback

In the standard reinforcement learning setting, for each action an agent takes, the environment provides a reward.
This is encoded by the function $R(s,a)$, where $s$ is a state and $a$ in an action.

In DAAF settings, the environment instead provides feedback at periodic time intervals (e.g. based on a Poisson distribution), and on aggregate, in the sense that the agent gets a combination of rewards for several actions.
The fact that the agent cannot discern how much each action taken contributes to the observed reward makes the feedback anonymous.

To constrast with fully sparse reward problems, where the reward is only observed at the end, after task completion or failure, DAAF problems have intermittent feedback.


## Code

Contains
  - Algorithms for policy control with DAAF
  - Algorithms for policy evaluation with DAAF
  - Notebooks with analysis results on reward rstimation or recovery


## Submissions

For specific snapshots of code submitted to conferences:

  1. [DS '22 - Policy Evaluation](docs/ds22.md)
  2. [EMCL-PKDD '24 - Policy Control](docs/ecmlpkdd24.md)


## Dev Env
First, make sure the following python development tools are installed:
  - [uv](https://docs.astral.sh/uv/getting-started/installation/)
  - [ruff](https://docs.astral.sh/ruff/installation/)

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

All requirements files are compiled using `uv`.