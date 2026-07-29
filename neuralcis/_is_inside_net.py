from ._simulator_net import _SimulatorNet
from ._param_sampling_net import _ParamSamplingNet
from ._utils import known_params_from_params
from .common import TESTING
from . import common

import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
import numpy as np

from typing import Callable, Tuple, Sequence
from .common import Params, KnownParams, Stats, Samples, NetInputs
import tensor_annotations.tensorflow as ttf
from tensor_annotations.tensorflow import Tensor1, Tensor2
tf32 = ttf.float32


NetInputBlob = Tuple[Tensor2[tf32, Samples, Stats],
                     Tensor2[tf32, Samples, KnownParams]]
NetTargetBlob = Tensor1[tf32, Samples]                 # 0
NetOutputBlob = Tensor1[tf32, Samples]                 # Is inside


class _IsInsideNet(_SimulatorNet):
    absolute_loss_increase_tol = common.ABS_LOSS_INCREASE_TOL_Z_NET
    smallest_profile_found_in = TESTING

    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],                # params
                Tensor2[tf32, Samples, Stats]                # -> estimates
            ],
            param_sampling_net: _ParamSamplingNet,
            preprocess_params_fn: Callable[
                [Tensor2[tf32, Samples, KnownParams], bool],
                Tensor2[tf32, Samples, KnownParams]
            ],
            num_stat: int,
            num_unknown_param: int,
            num_known_param: int,
            profile: str,
            **network_setup_args,
    ) -> None:

        super().__init__(
            profile,
            num_inputs_for_each_net=(num_stat + num_known_param,),
            num_outputs_for_each_net=(1,),
            instance_tf_variables_to_save=("stat_mins",
                                           "stat_maxs"),
            **network_setup_args
        )

        if self._skip_when_profile(profile):
            return

        self.sampling_distribution_fn = sampling_distribution_fn
        self.param_sampling_net = param_sampling_net
        self.preprocess_params_fn = preprocess_params_fn

        self.num_stat = num_stat
        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param

        assert self.batch_size % 2 == 0
        self.batch_size_inside = self.batch_size // 2
        self.batch_size_dummy = self.batch_size // 2

        self.stat_mins = tf.Variable(tf.fill(num_stat, np.nan))
        self.stat_maxs = tf.Variable(tf.fill(num_stat, np.nan))

    ###########################################################################
    #
    #  Methods overridden from _SimulatorNet
    #
    ###########################################################################

    def get_ready_for_training(self) -> None:
        params_for_sampling = self.sample_params(100000, preprocess=True)
        stats_init = self.sampling_distribution_fn(params_for_sampling)
        stat_mins = tfp.stats.percentile(stats_init, q=00.1, axis=0)
        stat_maxs = tfp.stats.percentile(stats_init, q=99.9, axis=0)

        self.stat_mins.assign(stat_mins)
        self.stat_maxs.assign(stat_maxs)

        super().get_ready_for_training()

    def simulate_training_data(
            self,
    ) -> Tuple[
            NetInputBlob,
            NetTargetBlob,
    ]:

        params = self.sample_params(self.batch_size_inside, preprocess=True)
        stats_inside = self.sampling_distribution_fn(params)
        known_params_inside = known_params_from_params(params,
                                                       self.num_unknown_param,
                                                       self.num_known_param)
        stats_shape_dummy = (self.batch_size_dummy, self.num_stat)
        stats_dummy = tf.random.uniform(stats_shape_dummy,
                                        minval=self.stat_mins[None, :],
                                        maxval=self.stat_maxs[None, :])
        known_param_shape_dummy = (self.batch_size_dummy, self.num_known_param)
        known_params_dummy = tf.random.uniform(known_param_shape_dummy,
                                               minval=common.PARAMS_MIN,
                                               maxval=common.PARAMS_MAX)
        known_params_dummy = self.preprocess_params_fn(known_params_dummy,
                                                       known_params_only=True)

        stats = tf.concat([stats_inside, stats_dummy], axis=0)
        known_params = tf.concat([known_params_inside, known_params_dummy],
                                 axis=0)

        input_blob = (stats, known_params)

        target_blob = tf.concat([tf.ones(self.batch_size_inside),
                                 tf.zeros(self.batch_size_dummy)], axis=0)

        return input_blob, target_blob

    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: NetTargetBlob,
    ) -> ttf.float32:

        return tf.reduce_mean(tf.square(net_outputs[:, 0] - target_outputs))

    def net_inputs(
            self,
            input_blob: NetInputBlob,
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], ...]:

        stats, known_params = input_blob
        input_tensor = tf.concat([stats, known_params], axis=1)
        net_inputs = (input_tensor,)
        return net_inputs

    ###########################################################################
    #
    #  Tensorflow members
    #
    ###########################################################################

    def sample_params(
            self,
            n: int,
            preprocess: bool,
    ) -> Tensor2[tf32, Samples, Params]:

        return self.param_sampling_net.sample_params(n, preprocess)

    def is_inside_sampled_region(
            self,
            stats: Tensor2[tf32, Samples, Stats],
            known_params: Tensor2[tf32, Samples, KnownParams],
    ) -> Tensor1[ttf.bool, Samples]:

        threshold = common.IS_INSIDE_NET_THRESHOLD
        net_says_inside = self.call_tf((stats, known_params)) > threshold
        bounds_say_stats_inside = (
            (stats >= self.stat_mins[None, :]) &
            (stats <= self.stat_maxs[None, :])
        )
        bounds_say_known_params_inside = (
            (known_params >= common.PARAMS_MIN) &
            (known_params <= common.PARAMS_MAX)
        )
        is_inside = (net_says_inside[:, 0] &
                     tf.reduce_all(bounds_say_stats_inside, axis=1) &
                     tf.reduce_all(bounds_say_known_params_inside, axis=1))

        return is_inside
