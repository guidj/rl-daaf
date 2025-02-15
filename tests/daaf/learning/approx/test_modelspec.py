import copy
from typing import Any, Callable, Tuple

import hypothesis
import numpy as np
from hypothesis import strategies as st

from daaf.learning.approx import modelspec


class ProviderValueFn(modelspec.ValueFnModel):
    """
    A value function based on a callable given
    at instantiation.
    """

    def __init__(self, value_fn: Callable[[Any], float]):
        self._value_fn = value_fn
        self._weights = None

    def predict(self, inputs: modelspec.ValueFnInputs) -> float:
        """
        Computes the value for a given context,
        i.e. state or (state, action)
        """
        return (
            0.0 if inputs.is_terminal_state is True else self._value_fn(inputs.features)
        )

    def gradients(self, inputs: modelspec.ValueFnInputs) -> Any:
        """
        Computes the gradients of the output
        with respect to the weights.
        """
        return inputs.features

    def predict_and_gradients(
        self, inputs: modelspec.ValueFnInputs
    ) -> Tuple[float, Any]:
        """
        Computes prediction and gradients of the
        weights for the prediction.
        """
        return self.predict(inputs), self.gradients(inputs)

    def weights(self) -> Any:
        """
        Returns the current weights of the model.
        """
        return self._weights

    def assign_weights(self, weights: Any) -> None:
        """
        Assigns new values to the model weights.
        """
        self._weights = copy.deepcopy(weights)


@hypothesis.given(observation=st.integers(min_value=-100, max_value=100))
def test_approx_fn_predict(observation: int):
    approx_fn = modelspec.ApproxFn(
        model=ProviderValueFn(create_state_values_fn()), pre_proc=pre_proc
    )
    # negative states are terminal - v(s) = 0
    assert (
        approx_fn.predict(observation=np.array(observation))
        == np.max([observation, 0.0]) ** 2
    )


@hypothesis.given(observation=st.integers(min_value=-100, max_value=100))
def test_approx_fn_gradients(observation: int):
    approx_fn = modelspec.ApproxFn(
        model=ProviderValueFn(create_state_values_fn()), pre_proc=pre_proc
    )
    assert approx_fn.gradients(observation=np.array(observation)) == observation


@hypothesis.given(observation=st.integers(min_value=-100, max_value=100))
def test_approx_fn_predict_and_gradients(observation: int):
    approx_fn = modelspec.ApproxFn(
        model=ProviderValueFn(create_state_values_fn()), pre_proc=pre_proc
    )
    # negative states are terminal - v(s) = 0
    assert approx_fn.predict_and_gradients(np.array(observation)) == (
        np.max([observation, 0.0]) ** 2,
        observation,
    )


def test_approx_fn_weights():
    approx_fn = modelspec.ApproxFn(
        model=ProviderValueFn(create_state_values_fn()), pre_proc=pre_proc
    )
    assert approx_fn.weights() is None
    approx_fn.assign_weights(0)
    assert approx_fn.weights() == 0
    approx_fn.assign_weights(100.0)
    assert approx_fn.weights() == 100.0


def pre_proc(observation: Any) -> modelspec.ValueFnInputs:
    """
    Negative states are terminal.
    """
    return modelspec.ValueFnInputs(
        features=observation, is_terminal_state=observation.item() < 0
    )


def create_state_values_fn() -> Callable[[Any], float]:
    """
    Static values for a hypothetical environment.
    """

    def value_fn(features: Any) -> float:
        value: float = features**2.0
        return value

    return value_fn
