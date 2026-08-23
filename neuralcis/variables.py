from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.eager.def_function import Function as TFFunction        # type: ignore

from .common import Samples
from . import common
from typing import Optional, Union, Tuple, Sequence
from tensor_annotations.tensorflow import Tensor0, Tensor1
from tensor_annotations.tensorflow import float32 as tf32
import tensor_annotations.tensorflow as ttf

AnyTensor = Union[Tensor0, Tensor1]


class VariableType(ABC):
    # If is a param, will be clipped to these limits
    param_hard_min_human = None
    param_hard_max_human = None
    accepts_lowish_highish_in_constructor = True
    axis_type = "linear"

    def __init__(
            self,
            # Default values here give no rescale.
            lowish_highish_values: Optional[Tuple[float, float]] = None,
    ) -> None:

        self.rescale = False
        if lowish_highish_values is not None:
            self.adjust_and_set_lowish_highish(lowish_highish_values)

    def adjust_and_set_lowish_highish(
            self,
            lowish_highish_values: Tuple[float, float]
    ) -> Tuple[float, float]:

        self.rescale = True
        lowish, highish = lowish_highish_values

        self.human_lowish = lowish
        self.human_highish = highish

        # Default version does not adjust the lowish_highish_values at all
        return lowish_highish_values

    @property
    def trans_param_lowish(self) -> Tensor0[tf32]:
        return self.to_net_transform_param(tf.constant(self.human_lowish))

    @property
    def trans_param_width(self) -> Tensor0[tf32]:
        highish = tf.constant(self.human_highish)
        trans_param_highish = self.to_net_transform_param(highish)
        return trans_param_highish - self.trans_param_lowish

    @property
    def trans_stat_lowish(self) -> Tensor0[tf32]:
        return self.to_net_transform_stat(tf.constant(self.human_lowish))

    @property
    def trans_stat_width(self) -> Tensor0[tf32]:
        highish = tf.constant(self.human_highish)
        trans_stat_highish = self.to_net_transform_stat(highish)
        return trans_stat_highish - self.trans_stat_lowish

    @property
    def net_lowish(self) -> Tensor0[tf32]:
        return tf.constant(common.PARAMS_MIN)

    @property
    def net_width(self) -> Tensor0[tf32]:
        return tf.constant(common.PARAMS_MAX - common.PARAMS_MIN)

    @abstractmethod
    def to_net_transform_generic(self, human):
        pass

    @abstractmethod
    def from_net_tranform_generic(self, transformed):
        pass

    def to_net_param(self, human):
        transformed = self.to_net_transform_param(human)
        if self.rescale:
            net = self.to_net_rescale(transformed, self.trans_param_lowish,
                                                   self.trans_param_width)
        else:
            net = transformed
        return net

    def to_net_stat(self, human):
        transformed = self.to_net_transform_stat(human)
        if self.rescale:
            net = self.to_net_rescale(transformed, self.trans_stat_lowish,
                                                   self.trans_stat_width)
        else:
            net = transformed
        return net

    def from_net_param(self, net):
        if self.rescale:
            transformed = self.from_net_rescale(net, self.trans_param_lowish,
                                                     self.trans_param_width)
        else:
            transformed = net
        human = self.from_net_transform_param(transformed)
        return human

    def from_net_stat(self, net):
        if self.rescale:
            transformed = self.from_net_rescale(net, self.trans_stat_lowish,
                                                     self.trans_stat_width)
        else:
            transformed = net
        human = self.from_net_transform_stat(transformed)
        return human

    def preprocess_params(
            self,
            params_human: Tensor1[tf32, Samples],  # Only done to params!
    ) -> Tensor1[tf32, Samples]:

        raise Exception('VariableType.process_params should not be run unless'
                        ' it has been explicitly implemented on a subclass.')

    def to_net_rescale(self, transformed, from_lowish, from_width):
        between_0_1 = (transformed - from_lowish) / from_width
        net = between_0_1 * self.net_width + self.net_lowish
        return net

    def from_net_rescale(self, net, to_lowish, to_width):
        between_0_1 = (net - self.net_lowish) / self.net_width
        transformed = between_0_1 * to_width + to_lowish
        return transformed

    def to_net_transform_param(self, human):
        return self.to_net_transform_generic(human)

    def to_net_transform_stat(self, human):
        return self.to_net_transform_generic(human)

    def from_net_transform_param(self, transformed):
        return self.from_net_tranform_generic(transformed)

    def from_net_transform_stat(self, transformed):
        return self.from_net_tranform_generic(transformed)


class Variable(ABC):
    exists_pre_canonicalization = True
    def __init__(
            self,
            vtype: VariableType,
            canonicalize: Optional[str] = None,
    ):

        self.vtype = vtype
        self.canonicalize_str = canonicalize
        self.canonicalize_fn = None
        self.canonicalize_my_name = None

    @abstractmethod
    def to_net(self, human):
        pass

    @abstractmethod
    def from_net(self, net):
        pass

    def has_canonicalize(self):
        return self.canonicalize_str is not None

    def render_canonicalize_fn(
            self,
            my_name: str,
            stat_names: Sequence[str],
    ) -> None:

        if not self.has_canonicalize():
            raise Exception(f"Missing canonicalize string in {my_name}!")

        stat_names_list = ", ".join(stat_names)
        if my_name not in stat_names and self.exists_pre_canonicalization:
            args_list = f"{my_name}, {stat_names_list}"
        else:
            args_list = stat_names_list

        canonicalize_fn_str = f"lambda {args_list}: {self.canonicalize_str}"
        canonicalize_fn = eval(canonicalize_fn_str, {"tf": tf, "tfp": tfp})
        self.canonicalize_fn = canonicalize_fn
        self.canonicalize_my_name = my_name

    def canonicalize(
            self,
            my_value_human: Optional[Tensor1[tf32, Samples]] = None,
            **stats_human: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        if self.has_canonicalize():
            kwargs = stats_human
            if self.exists_pre_canonicalization:
                assert my_value_human is not None
                kwargs |= {self.canonicalize_my_name: my_value_human}
            else:
                assert my_value_human is None
            return self.canonicalize_fn(**kwargs)
        else:
            raise Exception("No canonicalization setup for this variable!!")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["canonicalize_fn"] = None
        return state


###############################################################################
#  A Variable may be a Param, KnownParam, Stat, StatCanonical or Interest:
###############################################################################


class Param(Variable):
    is_known_param = False

    def __init__(
            self,
            vtype: VariableType,
            estimates_box_min_max: Tuple[float, float],
            canonicalize: Optional[str] = None,
    ) -> None:

        super().__init__(vtype, canonicalize)
        box_min_max_tf = tf.constant(estimates_box_min_max)
        self.estimates_box_min_and_max_net = self.to_net(box_min_max_tf)

        hard_min_human = self.vtype.param_hard_min_human
        hard_max_human = self.vtype.param_hard_max_human
        self.has_hard_limits = (hard_min_human is not None
                                or hard_max_human is not None)
        self.hard_min_net = self.__human_or_none_to_net(hard_min_human, -1.)
        self.hard_max_net = self.__human_or_none_to_net(hard_max_human, +1.)

    def __human_or_none_to_net(
            self,
            human_float: float,
            inf_sign: float,
    ) -> Tensor0[tf32]:

        if human_float is not None:
            return self.to_net(tf.constant(human_float))
        else:
            return tf.constant(inf_sign * np.inf)

    def to_net(
            self,
            human: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        return self.vtype.to_net_param(human)

    def from_net(
            self,
            net: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        return self.vtype.from_net_param(net)

    def preprocess_net_interface(
            self,
            net: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        # Only convert to human if there is actually an extra_process method
        if (type(self.vtype).preprocess_params
                is not VariableType.preprocess_params):
            human = self.from_net(net)
            human = self.vtype.preprocess_params(human)
            net = self.to_net(human)

        if self.has_hard_limits:
            net = tf.clip_by_value(net, self.hard_min_net, self.hard_max_net)

        return net

    def is_within_hard_limits(
            self,
            net: Tensor1[tf32, Samples],
    ) -> Tensor1[ttf.bool, Samples]:

        if self.has_hard_limits:
            return (net >= self.hard_min_net) & (net <= self.hard_max_net)
        else:
            return tf.ones_like(net, dtype=tf.bool)


class KnownParam(Param):
    is_known_param = True
    def __init__(
            self,
            vtype: VariableType,
            min_max: Tuple[float, float],
            canonicalize_str: Optional[str] = None,
    ) -> None:

        if vtype.accepts_lowish_highish_in_constructor and vtype.rescale:
            raise Exception("For a KnownParam your VariableType should NOT"
                            " have its lowish_highish_values preset!  They are"
                            " instead set through the required min_max "
                            " argument when constructing the KnownParam.")
        min_max_adj = vtype.adjust_and_set_lowish_highish(min_max)
        vtype.param_hard_min_human, vtype.param_hard_max_human = min_max_adj
        super().__init__(vtype, min_max_adj, canonicalize_str)


class Stat(Variable):
    def __init__(
            self,
            vtype: VariableType,
    ) -> None:

        super().__init__(vtype, canonicalize=None)

    def to_net(
            self,
            human: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        return self.vtype.to_net_stat(human)

    def from_net(
            self,
            net: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        return self.vtype.from_net_stat(net)


class StatCanonical(Variable):
    exists_pre_canonicalization = False                          # A canonical stat is freshly computed from the other stats, so cannot be used in its own computation
    def __init__(
            self,
            vtype: VariableType,
            canonicalize: str,
    ) -> None:

        super().__init__(vtype, canonicalize)

    def to_net(
            self,
            human: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        return self.vtype.to_net_stat(human)

    def from_net(
            self,
            net: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        return self.vtype.from_net_stat(net)


class Interest(Variable):
    def to_net(
            self,
            human: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        return self.vtype.to_net_param(human)

    def from_net(
            self,
            net: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        return self.vtype.from_net_param(net)


###############################################################################
#  Available variable types:
###############################################################################


class Location(VariableType):
    def to_net_transform_generic(self, human):
        return human

    def from_net_tranform_generic(self, transformed):
        return transformed


class Scale(VariableType):
    axis_type = "log"
    def to_net_transform_generic(self, human):
        return tf.math.log(human)

    def from_net_tranform_generic(self, transformed):
        return tf.math.exp(transformed)


class PositiveCount(Scale):
    def adjust_and_set_lowish_highish(
            self,
            lowish_highish_values: Tuple[float, float],
    ) -> Tuple[float, float]:

        lowish, highish = lowish_highish_values
        return super().adjust_and_set_lowish_highish((lowish - 0.5,
                                                      highish + 0.5))

    def preprocess_params(self, params_human):
        return tf.floor(params_human + 0.5)


class Correlation(Location):
    param_hard_min_human = -0.99
    param_hard_max_human = 0.99
    accepts_lowish_highish_in_constructor = False
    def __init__(self):
        super().__init__()

    def to_net_transform_stat(self, human):
        human_safe = human * 0.99
        return tf.math.atanh(human_safe)

    def from_net_transform_stat(self, transformed):
        human_safe = tf.math.tanh(transformed)
        human = human_safe / 0.99
        return human


class Proportion(Location):
    param_hard_min_human = 0.01
    param_hard_max_human = 0.99
    accepts_lowish_highish_in_constructor = False
    def __init__(self):
        super().__init__((0.1, 0.9))

    def to_net_transform_stat(self, human):
        human_safe = human * 0.9 + 0.05
        return self.logit(human_safe)

    def from_net_transform_stat(self, transformed):
        human_safe = tf.math.sigmoid(transformed)
        human = (human_safe - 0.05) / 0.9
        return human

    @staticmethod
    def logit(p):
        return tf.math.log(p) - tf.math.log1p(-p)
