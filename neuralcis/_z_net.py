from ._simulator_net import _SimulatorNet
from . import _utils, common

import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
import numpy as np

from typing import Callable, Tuple, Sequence, Optional, Union
from .common import Params, KnownParams, Stats, Zs, Samples
from .common import NetInputs
import tensor_annotations.tensorflow as ttf
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2
tf32 = ttf.float32


NetInputBlob = Tuple[Tensor2[tf32, Samples, Stats],                 # -> ys
                     Tensor2[tf32, Samples, Params]]                # -> params

NetTargetBlob = Tensor0

NetOutputBlob = Tuple[Tensor2[tf32, Samples, Zs],      # net outputs (z values)
                      Tensor1[tf32, Samples],          # Jacobian determinants
                      Tensor1[tf32, Samples]]          # dz0 / dinterest


class _ZNet(_SimulatorNet):
    absolute_loss_increase_tol = common.ABS_LOSS_INCREASE_TOL_Z_NET
    root2pi = tf.math.log(tf.math.sqrt(2 * np.pi))
    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],                 # params
                Tensor2[tf32, Samples, Stats]                     # -> ys
            ],
            param_sampling_fn: Callable[
                [int, int],                                       # n
                Tensor2[tf32, Samples, Params]                    # -> params
            ],
            interest_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor1[tf32, Samples]
            ],
            transform_on_stats_fn: Callable[
                [Tensor2[tf32, Samples, Stats],
                 Tensor2[tf32, Samples, Params]],
                Tuple[Tensor2[tf32, Samples, Stats],
                      Tensor2[tf32, Samples, Params]],
            ],
            num_stat: int,
            num_unknown_param: int,
            num_known_param: int,
            known_param_indices: Sequence[int],
            num_stats_remaining_after_transform: int,
            profile: str,
            **network_setup_args,
    ) -> None:

        # Allow MonotonicWithParams layers to be used on first net if needed.
        #  This counts all estimates except the first (num_y - 1) plus the
        #  interest (1) plus all known params as "not needing to be monotone"
        # TODO: Probably now don't need Monotonic Layers any more, so consider
        #       getting rid of this (but must remove monotonic layers at the
        #       same time).
        layer_kwargs = [
            {'num_params': (num_unknown_param - 1) + 1 + num_known_param},
            {},
        ]

        super().__init__(
            profile=profile,
            num_inputs_for_each_net=(
                num_stats_remaining_after_transform + 1 + num_known_param,
                num_stats_remaining_after_transform + num_unknown_param
                                                    + num_known_param
            ),
            num_outputs_for_each_net=(1, num_stat - 1),
            layer_kwargs=layer_kwargs,
            **network_setup_args
        )

        # TODO: Can probably reduce the redundancy here by only passing e.g.
        #       num_param and known_param_indices.  Can probably do that across
        #       all net types constructed by NeuralCIs to create a cleaner
        #       interface.
        assert len(known_param_indices) == num_known_param

        self.sampling_distribution_fn = sampling_distribution_fn
        self.param_sampling_fn = param_sampling_fn
        self.interest_fn = interest_fn
        self.transform_on_stats_fn = transform_on_stats_fn

        self.known_param_indices = known_param_indices

    ###########################################################################
    #
    #  Methods overridden from _SimulatorNet
    #
    ###########################################################################

    @tf.function
    def simulate_training_data(
            self,
    ) -> Tuple[
            NetInputBlob,
            NetTargetBlob,
    ]:

        n = self.batch_size
        no_target_data = tf.constant([[]], shape=(n, 0))
        return self.sample_stats_and_params(n), no_target_data

    @tf.function
    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: None = None,
    ) -> ttf.float32:

        outputs, jacobians, dz0_dinterest = net_outputs
        neg_log_likelihoods = self.neg_log_likelihoods(outputs, jacobians)
        dz0_dinterest_is_neg = tf.keras.activations.relu(dz0_dinterest)
        dz0_dinterest_penalty = \
            common.DZ0_DINTEREST_PENALTY_WEIGHT * dz0_dinterest_is_neg
        loss = tf.math.reduce_mean(neg_log_likelihoods + dz0_dinterest_penalty)

        tf.debugging.check_numerics(loss,
                                    "Na or inf in loss in multiple Z Net opt")

        return loss

    @tf.function
    def call_tf_training(
            self,
            input_blob: NetInputBlob,
    ) -> NetOutputBlob:

        out, det, dz0_dcon = self.net_outputs_and_transformation_jacobdets(
            input_blob,
            training=True,
        )
        return out, det, dz0_dcon

    @tf.function
    def net_inputs(
            self,
            input_blob: NetInputBlob,
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], ...]:

        # TODO: relate this to the interest rather than "known param" naming
        stats, params = input_blob
        stats_trans, params_trans = self.transform_on_stats_fn(stats, params)
        interest_trans = self.interest_fn(params_trans)

        known_params = tf.gather(params_trans,
                                 self.known_param_indices, axis=1)
        interest_net_inputs = tf.concat([stats_trans,
                                         interest_trans[:, None],
                                         known_params], axis=1)
        other_net_inputs = tf.concat([stats_trans,
                                      params_trans], axis=1)
        net_inputs = interest_net_inputs, other_net_inputs

        return net_inputs

    def compute_optimum_loss(self) -> ttf.float32:
        # TODO: the individual losses here are not currently saved.  Need to
        #       rewrite _DataSaver to have functions that can be overridden
        #       instead of taking constructor args which need screwing with.

        # TODO: implement optimum loss for multi-znet.

        return tf.constant(0.)

    ###########################################################################
    #
    #  Tensorflow members
    #
    ###########################################################################

    @tf.function
    def call_tf_interest_only(
            self,
            stats: Tensor2[tf32, Samples, Stats],
            interest: Tensor1[tf32, Samples],
            known_params: Tensor2[tf32, Samples, KnownParams],
    ) -> Tensor1[tf32, Samples]:

        raise Exception('call_tf_interest_only needs rewriting now that stats'
                        ' are transformed!!')

    @tf.function
    def neg_log_likelihoods(
            self,
            outputs: Tensor2[tf32, Samples, Zs],
            sample_jacobdets: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        eps = common.SMALLEST_LOGABLE_NUMBER

        jacobdets_floored = _utils._soft_floor_at_zero(sample_jacobdets)
        log_jacobdets_floored = tf.math.log(jacobdets_floored + eps)

        log_normal_pds = -0.5 * tf.square(outputs) - self.root2pi
        log_normal_pd_joint = tf.math.reduce_sum(log_normal_pds, axis=1)

        # work first additively in log space to avoid overflows
        neg_log_likelihoods = -log_normal_pd_joint - log_jacobdets_floored

        return neg_log_likelihoods

    @tf.function
    def net_outputs_and_transformation_jacobdets(
            self,
            input_blob: NetInputBlob,
            training=False,
    ) -> Tuple[
        Tensor2[tf32, Samples, Zs],                                            # Net outputs
        Tensor1[tf32, Samples],                                                # Jacobian determinants
        Tensor1[tf32, Samples],                                                # dz0 / dcontrast
    ]:

        stats, params = input_blob

        with tf.GradientTape(persistent=True) as tape:                         # type: ignore
            tape.watch(stats)
            tape.watch(params)
            input_blob = stats, params
            net_inputs = self.net_inputs(input_blob)
            zs = self._call_tf(net_inputs, training=training)
            z0 = zs[:, 0:1]

        # TODO: it seems to get stuck now trying to differentiate the Jacobian
        #       with pfor turned on, but didn't previously.  Need to fix.
        jacobians = tape.batch_jacobian(zs, stats, experimental_use_pfor=False)
        jacobdets = tf.linalg.det(jacobians)

        # Also compute dz0 / dinterest (UNTRANSFORMED INTEREST)
        dz0_dtheta = tape.batch_jacobian(z0, params,
                                         experimental_use_pfor=False)[:, 0, :]
        del tape

        with tf.GradientTape() as tape:
            tape.watch(params)
            interest_untransformed = self.interest_fn(params)[:, None]
        dcon_dtheta = tape.batch_jacobian(interest_untransformed, params,
                                          experimental_use_pfor=False)
        dcon_dtheta = dcon_dtheta[:, 0, :]

        dz0_dinterest = (tf.reduce_sum(dcon_dtheta * dz0_dtheta, axis=1)
                         / tf.reduce_sum(tf.square(dcon_dtheta), axis=1))

        return zs, jacobdets, dz0_dinterest                                    # type: ignore

    @tf.function
    def sample_params(
            self,
            n: int,
    ) -> Tensor2[tf32, Samples, Params]:

        return self.param_sampling_fn(0, n)

    @tf.function
    def sample_stats_and_params(
            self,
            n: int,
    ) -> NetInputBlob:

        params = self.sample_params(n)
        y = self.sampling_distribution_fn(params)
        return y, params

    @tf.function
    def z(
            self,
            stats: Tensor2[tf32, Samples, Stats],
            params: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        # We need a separate function for this, as we might not have our
        # params in the right format for the second net after a transform
        # on estimates call.
        net0_inputs, _ = self.net_inputs((stats, params))
        z = self.nets[0](net0_inputs)[:, 0]
        return z

    def fit(
            self,
            steps_per_epoch: int = common.STEPS_PER_EPOCH_ZNET,
            epochs: int = common.EPOCHS_ZNET,
            *args, **kwargs,
    ):

        return super().fit(steps_per_epoch, epochs, *args, **kwargs)
