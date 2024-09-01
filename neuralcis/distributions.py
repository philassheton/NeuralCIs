from abc import ABC, abstractmethod
import tensorflow as tf

from typing import Optional, Union, Tuple
from neuralcis.common import Samples, MinAndMax
from tensor_annotations.tensorflow import Tensor0, Tensor1
from tensor_annotations.tensorflow import float32 as tf32

AnyTensor = Union[Tensor0, Tensor1]


class Distribution(ABC):
    axis_type = "linear"

    def __init__(
            self,
            estimate_min: float,
            estimate_max: float,
    ) -> None:

        self.estimate_min = estimate_min
        self.estimate_max = estimate_max
        min_and_max = tf.constant([estimate_min, estimate_max])
        self.min_and_max_std_uniform = self.to_std_uniform(min_and_max)

    @abstractmethod
    def to_std_uniform(
            self,
            params_tensor: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:
        pass

    @abstractmethod
    def from_std_uniform(
            self,
            std_uniform_tensor: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:
        pass


class TransformUniformDistribution(Distribution):
    uniform_min: Tensor0
    uniform_max: Tensor0

    def __init__(
            self,
            min_value: float,
            max_value: float,
            estimate_min: Optional[float] = None,
            estimate_max: Optional[float] = None,
    ):

        assert max_value > min_value
        self.uniform_min = self.to_uniform_mapping(tf.constant(min_value))
        self.uniform_max = self.to_uniform_mapping(tf.constant(max_value))

        if estimate_min is None:
            estimate_min = min_value
        if estimate_max is None:
            estimate_max = max_value
        super().__init__(estimate_min, estimate_max)

    @abstractmethod
    def to_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        pass

    @abstractmethod
    def from_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        pass

    @tf.function
    def to_std_uniform(
            self,
            params_tensor: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        umin = self.uniform_min
        umax = self.uniform_max

        uniform = self.to_uniform_mapping(params_tensor)
        std_uniform = (uniform - umin) / (umax - umin)

        return std_uniform

    @tf.function
    def from_std_uniform(
            self,
            std_uniform_tensor: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        umin = self.uniform_min
        umax = self.uniform_max

        uniform = std_uniform_tensor * (umax - umin) + umin
        params = self.from_uniform_mapping(uniform)

        return params


class LogUniform(TransformUniformDistribution):
    axis_type = "log"

    def to_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        return tf.math.log(x)

    def from_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        return tf.math.exp(x)


class Uniform(TransformUniformDistribution):
    def to_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        return x

    def from_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        return x
