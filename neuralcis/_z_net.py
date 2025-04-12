from neuralcis._simulator_net import _SimulatorNet
from neuralcis import _utils
from neuralcis import common

import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
import numpy as np

from typing import Callable, Tuple, Sequence, Optional
from neuralcis.common import Params, KnownParams, Ys, Zs, Samples
from neuralcis.common import NetInputs, NetOutputs
import tensor_annotations.tensorflow as ttf
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2
tf32 = ttf.float32


NetInputBlob = Tuple[Tensor2[tf32, Samples, Ys],                    # -> ys
                     Tensor2[tf32, Samples, Params]]                # -> params

NetTargetBlob = Tensor0

NetOutputBlob = Tuple[Tensor2[tf32, Samples, Zs],      # net outputs (z values)
                      Tensor1[tf32, Samples],          # Jacobian determinants
                      Tensor1[tf32, Samples],          # dz0 / dcontrast
                      Tensor1[tf32, Samples]]          # estimated mean abs z0


class _ZNet(_SimulatorNet):
    absolute_loss_increase_tol = common.ABS_LOSS_INCREASE_TOL_Z_NET
    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],                 # params
                Tensor2[tf32, Samples, Ys]                        # -> ys
            ],
            param_sampling_fn: Callable[
                [int, int],                                       # n
                Tensor2[tf32, Samples, Params]                    # -> params
            ],
            contrast_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor1[tf32, Samples]
            ],
            transform_on_params_fn: Callable[
                [Tensor2[tf32, Samples, Ys],
                 Tensor2[tf32, Samples, Params]],
                Tuple[Tensor2[tf32, Samples, Ys],
                      Tensor2[tf32, Samples, Params]]
            ],
            num_unknown_param: int,
            num_known_param: int,
            known_param_indices: Sequence[int],
            num_params_remaining_after_transform: int,
            **network_setup_args,
    ) -> None:

        # Allow MonotonicWithParams layers to be used on first net if needed.
        #  This counts all estimates except the first (num_y - 1) plus the
        #  contrast (1) plus all known params as "not needing to be monotone"
        # TODO: Probably now don't need Monotonic Layers any more, so consider
        #       getting rid of this (but must remove monotonic layers at the
        #       same time).
        layer_kwargs = [
            {'num_params': (num_unknown_param - 1) + 1 + num_known_param},
            {},
            {},
        ]
        num_estimate = num_unknown_param
        num_param = num_unknown_param + num_known_param

        super().__init__(
            num_inputs_for_each_net=(num_estimate + 1 + num_known_param,
                                     num_estimate +
                                     num_params_remaining_after_transform,
                                     num_param),
            num_outputs_for_each_net=(1, num_estimate - 1, 1),
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
        self.contrast_fn = contrast_fn
        self.transform_on_params_fn = transform_on_params_fn

        self.num_estimate = num_estimate
        self.known_param_indices = known_param_indices



        # PHIL!!  Should this maybe be saved so we can always start from where
        #         we left off?
        self.step_counter = tf.Variable(0, dtype=tf.int64)

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
        return self.sample_ys_and_params(n), no_target_data

    @tf.function
    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: None = None,
    ) -> ttf.float32:

        zs, jacobians, dz0_dcontrast, mean_abs_z0 = net_outputs
        neg_log_likelihoods = self.neg_log_likelihoods(zs, jacobians)
        dz0_dcontrast_is_neg = 100. * tf.keras.activations.relu(dz0_dcontrast)
        losses_z = neg_log_likelihoods + dz0_dcontrast_is_neg

        abs_z = tf.stop_gradient(tf.math.abs(zs[:, 0]))
        losses_mean_abs_z = tf.square(abs_z - mean_abs_z0)

        mean_abs_weighting = self.get_mean_abs_weighting(mean_abs_z0)
        weight = tf.stop_gradient(mean_abs_weighting)

        loss = tf.math.reduce_mean(weight * losses_z + losses_mean_abs_z)

        tf.debugging.check_numerics(loss,
                                    "Na or inf in loss in multiple Z Net opt")

        return loss

    @tf.function
    def call_tf_training(
            self,
            input_blob: NetInputBlob,
    ) -> NetOutputBlob:

        out, det, dz0_dcon, mean_abs_z0 = \
            self.net_outputs_and_transformation_jacobdets(input_blob,
                                                          training=True)
        return out, det, dz0_dcon, mean_abs_z0

    @tf.function
    def net_inputs(
            self,
            input_blob: NetInputBlob,
            transform: bool = False,
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], ...]:

        # TODO: relate this to the contrast rather than "known param" naming
        ys, params = input_blob
        contrast = self.contrast_fn(params)
        return self.net_inputs_from_contrast(contrast, ys, params, transform)

    @tf.function
    def train_step(
            self,
            data,
    ):

        self.step_counter.assign(self.step_counter + 1)
        return super().train_step(data)

    ###########################################################################
    #
    #  Tensorflow members
    #
    ###########################################################################

    @tf.function
    def get_mean_abs_weighting(
            self,
            mean_abs_z0: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        goal = tf.math.log(tf.math.sqrt(2/np.pi))


        # PHIL! Neaten this up..  Shouldn't be logging then exping then log
        mean_abs_z0 = tf.minimum(mean_abs_z0, tf.math.exp(goal) * 1.1)
        mean_abs_z0 = tf.maximum(mean_abs_z0, tf.math.exp(goal) / 1.1)


        error = tf.square(tf.math.log(mean_abs_z0) - goal)
        steps_float = tf.cast(self.step_counter, tf.float32)
        step_weight = tf.sigmoid(steps_float / 10000. - 6.)



        squish_weighting = 1.



        return 1000. * squish_weighting * error * step_weight + 1.

    @tf.function
    def call_tf_contrast_only(
            self,
            estimates: Tensor2[tf32, Samples, Ys],                             # TODO: Swap Ys for Estimates.  We don't need Ys any more
            contrast: Tensor1[tf32, Samples],
            known_params: Tensor2[tf32, Samples, KnownParams],
    ) -> Tensor1[tf32, Samples]:

        # TODO: Might want to clean this up to make it more idiomatic
        net0_inputs = tf.concat([estimates, contrast[:, None], known_params],
                                axis=1)
        z = self.nets[0](net0_inputs)
        return z[:, 0]

    @tf.function
    def call_tf_transformed(
            self,
            input_blob: NetInputBlob,
    ) -> Tensor2[tf32, Samples, NetOutputs]:

        # This is only for use in the p_workings function to analyse the net.
        #  -- NB This is ONLY needed if we are accessing the second z-net,
        #     since at inference time the first should be transformed on
        #     estimates which is done in the neuralcis object
        # TODO: must be a cleaner way -- eg. transform input_blob directly
        net_inputs = self.net_inputs(input_blob, transform=True)
        return self._call_tf(net_inputs, training=False)

    @tf.function
    def net_inputs_from_contrast(
            self,
            contrast: Tensor1[tf32, Samples],
            ys: Tensor2[tf32, Samples, Ys],
            params: Tensor2[tf32, Samples, Params],
            transform: bool = False,
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], ...]:

        if transform:
            ys_trans, params_trans = self.transform_on_params_fn(ys, params)
        else:
            ys_trans, params_trans = ys, params

        contrast = contrast[:, None]
        known_params = tf.gather(params, self.known_param_indices, axis=1)
        contrast_net_inputs = tf.concat([ys, contrast, known_params],
                                        axis=1)
        other_net_inputs = tf.concat([ys_trans, params_trans], axis=1)
        return contrast_net_inputs, other_net_inputs, params

    @tf.function
    def neg_log_likelihoods(
            self,
            zs: Tensor2[tf32, Samples, Zs],
            sample_jacobdets: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        eps = common.SMALLEST_LOGABLE_NUMBER

        jacobdets_floored = _utils._soft_floor_at_zero(sample_jacobdets)

        normal_pd = tfp.distributions.Normal(0.0, 1.0).prob(zs)
        normal_pd_joint = tf.math.reduce_prod(normal_pd, axis=1) + eps

        # work first additively in log space to avoid overflows
        neg_log_likelihoods = (-tf.math.log(normal_pd_joint)
                               - tf.math.log(jacobdets_floored + eps))

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
        Tensor1[tf32, Samples],
    ]:

        ys, params = input_blob

        with tf.GradientTape(persistent=True) as tape:                         # type: ignore
            tape.watch(ys)
            contrast = self.contrast_fn(params)
            tape.watch(contrast)

            inputs = self.net_inputs_from_contrast(contrast, ys, params,
                                                   transform=True)
            outputs = self._call_tf(inputs, training=training)


            # PHIL!!  Instead of splitting here,
            #         refactor so that we can override all three outputs being
            #         stitched together!  (NB you already got rid of the other
            #         split!!)
            zs, mean_abs_z0 = tf.split(outputs, (self.num_estimate, 1), axis=1)


            z0 = zs[:, 0]

        dz0_dcontrast = tape.gradient(z0, contrast)
        jacobians = tape.batch_jacobian(zs, ys)
        jacobdets = tf.linalg.det(jacobians)
        del tape

        return zs, jacobdets, dz0_dcontrast, mean_abs_z0[:, 0]                 # type: ignore

    @tf.function
    def sample_params(
            self,
            n: int,
    ) -> Tensor2[tf32, Samples, Params]:

        assert n % 2 == 0
        n_outer = n // 2
        n_inner = n // 2
        return self.param_sampling_fn(n_inner, n_outer)

    @tf.function
    def sample_ys_and_params(
            self,
            n: int,
    ) -> NetInputBlob:

        params = self.sample_params(n)
        y = self.sampling_distribution_fn(params)
        return y, params

    @tf.function
    def z(
            self,
            ys: Tensor2[tf32, Samples, Ys],
            params: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        # We need a separate function for this, as we might not have our
        # params in the right format for the second net after a transform
        # on estimates call.
        net0_inputs, _, _ = self.net_inputs((ys, params))
        z = self.nets[0](net0_inputs)[:, 0]
        return z
