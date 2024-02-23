"""
Experiment constants.
"""


IDENTITY_MAPPER = "identity-mapper"
DAAF_AVERAGE_REWARD_MAPPER = "daaf-average-reward-mapper"
DAAF_IMPUTE_REWARD_MAPPER = "daaf-impute-missing-reward-mapper"
DAAF_LSQ_REWARD_ATTRIBUTION_MAPPER = "daaf-lsq-reward-attribution-mapper"
MDP_WITH_OPTIONS_MAPPER = "mdp-with-options-mapper"
DAAF_NSTEP_TD_UPDATE_MARK_MAPPER = "daaf-nstep-td-update-mark-mapper"
AGGREGATE_MAPPER_METHODS = (
    IDENTITY_MAPPER,
    DAAF_AVERAGE_REWARD_MAPPER,
    DAAF_IMPUTE_REWARD_MAPPER,
    DAAF_LSQ_REWARD_ATTRIBUTION_MAPPER,
    MDP_WITH_OPTIONS_MAPPER,
    DAAF_NSTEP_TD_UPDATE_MARK_MAPPER,
)
OPTIONS_POLICY = "options"
SINGLE_STEP_POLICY = "single-step"

DEFAULT_BUFFER_SIZE_MULTIPLIER = 2**10

SARSA = "sarsa"
ONE_STEP_TD = "one-step-td"
FIRST_VISIT_MONTE_CARLO = "first-visit-mc"
NSTEP_TD = "nstep-td"
ALGORITHMS = (ONE_STEP_TD, FIRST_VISIT_MONTE_CARLO, NSTEP_TD)
