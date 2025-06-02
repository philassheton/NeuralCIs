from ._simulator_net import _SimulatorNet
from ._param_sampling_net import _ParamSamplingNet
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
            num_unknown_param: int,
            num_known_param: int,
            known_param_indices: Sequence[int],
            profile: str,
            **network_setup_args,
    ) -> None:

        num_estimate = num_unknown_param

        super().__init__(
            profile,
            num_inputs_for_each_net=(num_estimate + num_known_param,),
            num_outputs_for_each_net=(1,),
            instance_tf_variables_to_save=("estimate_mins",
                                           "estimate_maxs",
                                           "known_param_mins",
                                           "known_param_maxs"),
            **network_setup_args
        )

        if self._skip_when_profile(profile):
            return

        self.sampling_distribution_fn = sampling_distribution_fn
        self.param_sampling_net = param_sampling_net
        self.preprocess_params_fn = preprocess_params_fn
        self.known_param_indices = known_param_indices

        self.num_estimate = num_estimate
        self.num_known_param = num_known_param

        assert self.batch_size % 2 == 0
        self.batch_size_inside = self.batch_size // 2
        self.batch_size_dummy = self.batch_size // 2

        self.estimate_mins = tf.Variable(tf.fill(num_estimate, np.nan))
        self.estimate_maxs = tf.Variable(tf.fill(num_estimate, np.nan))
        self.known_param_mins = tf.Variable(tf.fill(num_known_param, np.nan))
        self.known_param_maxs = tf.Variable(tf.fill(num_known_param, np.nan))

    ###########################################################################
    #
    #  Methods overridden from _SimulatorNet
    #
    ###########################################################################

    def get_ready_for_training(self) -> None:

        # We want to make sure there is a proportionate amount of the known
        # params at each end of the spectrum.  Therefore, we have to base
        # the mins and maxs from which we will draw dummy params on the
        # NON preprocessed values.  Then we need to make sure they are all
        # preprocessed before being sent to the net.
        #
        # This should not cause any problem during inference time, as all
        # params sent to the net should be preprocessed, meaning that they
        # cannot go beyond those preprocessed values, so the gap between the
        # preprocessed params and the boundary from the non-preprocessed
        # params will never be hit.
        #
        # TODO: Although the above is true, should consider preprocessing the
        #       mins and maxs AFTER TRAINING (or in a separate variable even)
        #       so that we get a robust boundary for inference time.
        params_init = self.sample_params(100000, preprocess=False)
        estimates_init = self.sampling_distribution_fn(params_init)
        estimate_mins = tfp.stats.percentile(estimates_init, q=00.1, axis=0)
        estimate_maxs = tfp.stats.percentile(estimates_init, q=99.9, axis=0)

        param_mins = tf.reduce_min(params_init, axis=0)
        param_maxs = tf.reduce_max(params_init, axis=0)
        known_param_mins = tf.gather(param_mins, self.known_param_indices)
        known_param_maxs = tf.gather(param_maxs, self.known_param_indices)

        self.estimate_mins.assign(estimate_mins)
        self.estimate_maxs.assign(estimate_maxs)
        self.known_param_mins.assign(known_param_mins)
        self.known_param_maxs.assign(known_param_maxs)

        super().get_ready_for_training()

    @tf.function
    def simulate_training_data(
            self,
    ) -> Tuple[
            NetInputBlob,
            NetTargetBlob,
    ]:

        params = self.sample_params(self.batch_size_inside,
                                    preprocess=True)
        estimates_inside = self.sampling_distribution_fn(params)
        known_params_inside = tf.gather(params,
                                        self.known_param_indices,
                                        axis=1)
        estimates_shape_dummy = (self.batch_size_dummy, self.num_estimate)
        estimates_dummy = tf.random.uniform(estimates_shape_dummy,
                                            minval=self.estimate_mins[None, :],
                                            maxval=self.estimate_maxs[None, :])
        known_param_shape_dummy = (self.batch_size_dummy, self.num_known_param)
        known_params_dummy = tf.random.uniform(known_param_shape_dummy,
                                               minval=self.known_param_mins,
                                               maxval=self.known_param_maxs)
        known_params_dummy = self.preprocess_params_fn(known_params_dummy,
                                                       known_params_only=True)

        estimates = tf.concat([estimates_inside, estimates_dummy], axis=0)
        known_params = tf.concat([known_params_inside, known_params_dummy],
                                 axis=0)

        input_blob = (estimates, known_params)

        target_blob = tf.concat([tf.ones(self.batch_size_inside),
                                 tf.zeros(self.batch_size_dummy)], axis=0)

        return input_blob, target_blob

    @tf.function
    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: NetTargetBlob,
    ) -> ttf.float32:

        return tf.reduce_mean(tf.square(net_outputs[:, 0] - target_outputs))

    @tf.function
    def net_inputs(
            self,
            input_blob: NetInputBlob,
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], ...]:

        estimates, known_params = input_blob
        input_tensor = tf.concat([estimates, known_params], axis=1)
        net_inputs = (input_tensor,)
        return net_inputs

    ###########################################################################
    #
    #  Tensorflow members
    #
    ###########################################################################

    @tf.function
    def sample_params(
            self,
            n: int,
            preprocess: bool,
    ) -> Tensor2[tf32, Samples, Params]:

        return self.param_sampling_net.sample_params(n, preprocess)

    @tf.function
    def is_inside_sampled_region(
            self,
            estimates: Tensor2[tf32, Samples, Stats],
            known_params: Tensor2[tf32, Samples, KnownParams],
    ) -> Tensor1[ttf.bool, Samples]:

        threshold = common.IS_INSIDE_NET_THRESHOLD
        net_says_inside = self.call_tf((estimates, known_params)) > threshold
        bounds_say_estimates_inside = (
            (estimates >= self.estimate_mins[None, :]) &
            (estimates <= self.estimate_maxs[None, :])
        )
        bounds_say_known_params_inside = (
            (known_params >= self.known_param_mins[None, :]) &
            (known_params <= self.known_param_maxs[None, :])
        )
        is_inside = (net_says_inside[:, 0] &
                     tf.reduce_all(bounds_say_estimates_inside, axis=1) &
                     tf.reduce_all(bounds_say_known_params_inside, axis=1))

        return is_inside
