from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.python.eager.def_function import Function as TFFunction        # type: ignore

from typing import Optional, Union
from .common import Samples
from . import sampling, common
from tensor_annotations.tensorflow import Tensor0, Tensor1
from tensor_annotations.tensorflow import float32 as tf32
import tensor_annotations.tensorflow as ttf

AnyTensor = Union[Tensor0, Tensor1]


class Variable(ABC):
    axis_type = "linear"

    param_hard_min_human = None
    param_hard_max_human = None

    def __init__(
            self,
            min: float,
            max: float,
    ) -> None:

        self.min = min
        self.max = max
        min_and_max = tf.constant([min, max])
        self.min_and_max_std_uniform = self.to_std_uniform(min_and_max)

        self.param_hard_limits = None
        self.param_hard_min_net = None
        self.param_hard_max_net = None
        self.set_param_hard_min_max_net(self.param_hard_min_human,
                                        self.param_hard_max_human)

        self.is_known_param = False

        self.tf_function_methods = None
        self.track_tf_functions()

    def set_param_hard_min_max_net(
            self,
            hard_min_human: Optional[float],
            hard_max_human: Optional[float],
   ) -> None:

        if hard_min_human is None and hard_max_human is None:
            self.param_hard_limits = False
        else:
            self.param_hard_limits = True

            if hard_min_human is not None:
                self.param_hard_min_net = self.to_net(
                    tf.constant(hard_min_human)
                )
            else:
                self.param_hard_min_net = tf.constant(-np.inf)

            if hard_max_human is not None:
                self.param_hard_max_net = self.to_net(
                    tf.constant(hard_max_human)
                )
            else:
                self.param_hard_max_net = tf.constant(np.inf)

    @abstractmethod
    def to_std_uniform(
            self,
            values_human: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:
        pass

    @abstractmethod
    def from_std_uniform(
            self,
            std_uniform_tensor: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:
        pass

    @tf.function
    def from_net(
            self,
            values_net: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        std_uniform = self.net_to_std_uniform(values_net)
        values_human = self.from_std_uniform(std_uniform)
        return values_human

    @tf.function
    def to_net(
            self,
            values_human: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        std_uniform = self.to_std_uniform(values_human)
        return self.std_uniform_to_net(std_uniform)

    @tf.function
    def net_to_std_uniform(
            self,
            values_net: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        return sampling.uniform_to_std_uniform(
            values_net, common.PARAMS_MIN, common.PARAMS_MAX
        )

    @tf.function
    def std_uniform_to_net(
            self,
            std_uniform: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        return sampling.uniform_from_std_uniform(
            std_uniform, common.PARAMS_MIN, common.PARAMS_MAX
        )

    @tf.function
    def is_valid_param(
            self,
            params_net: Tensor1[tf32, Samples],
    ) -> Tensor1[ttf.bool, Samples]:

        if self.param_hard_limits:
            return ((params_net >= self.param_hard_min_net) &
                    (params_net <= self.param_hard_max_net))
        else:
            return tf.ones_like(params_net, dtype=tf.bool)

    def from_std_uniform_valid_estimates(
            self,
            std_uniform: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        """Useful for random sampling from estimates box."""

        estimate_min_std_unif = self.to_std_uniform(self.min)
        estimate_max_std_unif = self.to_std_uniform(self.max)
        estimate_range_std_unif = estimate_max_std_unif - estimate_min_std_unif
        std_uniform = (std_uniform * estimate_range_std_unif
                       + estimate_min_std_unif)
        return self.from_std_uniform(std_uniform)


    @tf.function
    def preprocess(
            self,
            params_net: Tensor1[tf32, Samples],  # Only called on params!!
    ) -> Tensor1[tf32, Samples]:

        # Only convert to human if there is actually an extra_process method
        if type(self).extra_preprocess is not Variable.extra_preprocess:
            params_human = self.from_net(params_net)
            params_human = self.extra_preprocess(params_human)
            params_net = self.to_net(params_human)

        if self.param_hard_limits:
            params_net = tf.clip_by_value(params_net, self.param_hard_min_net,
                                                      self.param_hard_max_net)

        return params_net

    @tf.function
    def extra_preprocess(
            self,
            params_human: Tensor1[tf32, Samples],  # Only done to params!
    ) -> Tensor1[tf32, Samples]:

        raise Exception('Variable.extra_process should not be run unless it has'
                        ' been explicitly implemented on a subclass.')

    def make_known_param(self):
        self.is_known_param = True
        self.param_hard_limits = True
        self.set_param_hard_min_max_net(self.min, self.max)

    def track_tf_functions(self):
        self.tf_function_methods = []
        for attr_name in dir(self):
            func = getattr(self, attr_name)
            if isinstance(func, TFFunction):
                self.tf_function_methods.append(attr_name)

    def reapply_tf_functions(self):
        for method_name in self.tf_function_methods:
            method = getattr(self, method_name)
            setattr(self, method_name, tf.function(method))

    # When saving, we need to put the old Python methods back in place!!
    def deapply_tf_functions(self):
        for method_name in self.tf_function_methods:
            if method_name in self.__dict__:
                delattr(self, method_name)


class TransformUniformVariable(Variable):
    uniform_min: Tensor0
    uniform_max: Tensor0

    def __init__(
            self,
            min_value: float,
            max_value: float,
            # TODO: Naming needs overhaul; distinguish roles of various min/max
            min: Optional[float] = None,
            max: Optional[float] = None,
    ):

        assert max_value > min_value
        self.uniform_min = self.to_uniform_mapping(tf.constant(min_value))
        self.uniform_max = self.to_uniform_mapping(tf.constant(max_value))

        if min is None:
            min = min_value
        if max is None:
            max = max_value
        super().__init__(min, max)

    @abstractmethod
    def to_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        pass

    @abstractmethod
    def from_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        pass

    @tf.function
    def to_std_uniform(
            self,
            values_human: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        umin = self.uniform_min
        umax = self.uniform_max

        uniform = self.to_uniform_mapping(values_human)
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


class LogUniform(TransformUniformVariable):
    axis_type = "log"

    @tf.function
    def to_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        return tf.math.log(x)                                                  # type: ignore

    @tf.function
    def from_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        return tf.math.exp(x)                                                  # type: ignore


class Uniform(TransformUniformVariable):
    @tf.function
    def to_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        return x

    @tf.function
    def from_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        return x


class PositiveCount(LogUniform):
    @tf.function
    def extra_preprocess(
            self,
            params_human: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        return tf.floor(params_human + 0.5)


class SampleSize(PositiveCount):
    def __init__(
            self,
            min_value: int,
            max_value: int,
    ) -> None:

        super().__init__(min_value - 0.5, max_value + 0.5)

    @tf.function
    def to_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        return tf.math.log(x)                                                  # type: ignore

    @tf.function
    def from_uniform_mapping(self, x: AnyTensor) -> AnyTensor:
        return tf.math.exp(x)                                                  # type: ignore


class Correlation(Uniform):
    param_hard_min_human = -0.99
    param_hard_max_human = 0.99
    def __init__(self, min=-.99, max=.99):
        super().__init__(min, max)


class Proportion(Uniform):
    param_hard_min_human = 0.01
    param_hard_max_human = 0.99
    def __init__(self, min=0., max=1.):
        super().__init__(min, max)


Scale = LogUniform
Location = Uniform
