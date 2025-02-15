"""
This module constaints APIs and support
classes for function approximation in RL.
"""

import abc
import dataclasses
from typing import Any, Callable, Tuple


@dataclasses.dataclass(frozen=True)
class ValueFnInputs:
    """
    This class holds inputs for an approximation function.
    """

    features: Any
    is_terminal_state: bool


class ValueFnModel(abc.ABC):
    """
    This class defines an API for value functions
    using approximate solutions.

    Models should be differentiable.
    """

    @abc.abstractmethod
    def predict(self, inputs: ValueFnInputs) -> float:
        """
        Computes the value for a given context,
        i.e. state or (state, action)
        """

    @abc.abstractmethod
    def gradients(self, inputs: ValueFnInputs) -> Any:
        """
        Computes the gradients of the output
        with respect to the weights.
        """

    @abc.abstractmethod
    def predict_and_gradients(self, inputs: ValueFnInputs) -> Tuple[float, Any]:
        """
        Computes prediction and gradients of the
        weights for the prediction.
        """

    @abc.abstractmethod
    def weights(self) -> Any:
        """
        Returns the current weights of the model.
        """

    @abc.abstractmethod
    def assign_weights(self, weights: Any) -> None:
        """
        Assigns new values to the model weights.
        """


class ApproxFn:
    """
    Approximation of state or (state, action) functions.
    Wraps calls to a `ValueFnModel` by applying
    a transform operation to an observation
    (e.g. feature data encoding, normalisation).
    """

    def __init__(
        self,
        model: ValueFnModel,
        pre_proc: Callable[[Any], ValueFnInputs],
    ):
        """
        Initialises instance.
        """
        self._model = model
        self._pre_proc = pre_proc

    def predict(self, observation: Any) -> float:
        """
        Computes the value for a given context,
        i.e. state or (state, action)
        """
        return self._model.predict(self._pre_proc(observation))

    def gradients(self, observation: Any) -> Any:
        """
        Computes the gradients of the output
        with respect to the weights.
        """
        return self._model.gradients(self._pre_proc(observation))

    def predict_and_gradients(self, observation: Any) -> Tuple[float, Any]:
        """
        Computes prediction and gradients of the
        weights for the prediction.
        """
        return self._model.predict_and_gradients(self._pre_proc(observation))

    def weights(self) -> Any:
        """
        Returns the current weights of the model.
        """
        return self._model.weights()

    def assign_weights(self, weights: Any) -> None:
        """
        Assigns new values to the model weights.
        """
        self._model.assign_weights(weights)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model: {self._model})"
