from neuralcis._simulator_net import _SimulatorNet
from neuralcis import _utils
from neuralcis import common

import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore

from typing import Callable, Tuple, Sequence
from neuralcis.common import Params, Ys, Zs, Samples, NetInputs
import tensor_annotations.tensorflow as ttf
from tensor_annotations.tensorflow import Tensor1, Tensor2
tf32 = ttf.float32


NetInputBlob = Tuple[Tensor2[tf32, Samples, Ys],                    # -> ys
                     Tensor2[tf32, Samples, Params]]                # -> params

NetOutputBlob = Tuple[Tensor2[tf32, Samples, Zs],      # net outputs (z values)
                      Tensor1[tf32, Samples],          # Jacobian determinants
                      Tensor1[tf32, Samples]]          # dz0 / dcontrast


class _ZNet(_SimulatorNet):
    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],                 # params
                Tensor2[tf32, Samples, Ys]                        # -> ys
            ],
            param_sampling_fn: Callable[
                [int],                                            # n
                Tensor2[tf32, Samples, Params]                    # -> params
            ],
            contrast_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor1[tf32, Samples]
            ],
            validation_set_fn: Callable[
                [],
                NetInputBlob
            ],
            known_param_indices: Sequence[int],
            coords_fn: Callable[                                               # coords_fn allows us to remap the ys before they are fed into the znet
                [Tensor2[tf32, Samples, Ys]],                                  #   -- transformation Jacobians will then reach through this transform
                Tensor2[tf32, Samples, Ys]                                     #      to give the distribution of the *inputs* to this function.
            ] = tf.identity,
            **network_setup_args,
    ) -> None:

        self.sampling_distribution_fn = sampling_distribution_fn
        self.validation_set_fn = validation_set_fn
        self.param_sampling_fn = param_sampling_fn
        self.contrast_fn = contrast_fn
        self.coords_fn = coords_fn

        ys, params = self.validation_set_fn()

        num_y = tf.shape(ys).numpy()[1]
        self.num_z = [1, num_y - 1]

        # TODO:  Having to assign None here is a hack, since otherwise it
        #        insists on constructing the Model superclass first.  Under
        #        the current design, however, the bottom level class needs
        #        a few things in place before it can construct the
        #        _SimulatorNet superclass, which then constructs the
        #        Model superclass.  Needs a redesign to allow for construction
        #        of _SimulatorNet first  (e.g. by making the "build" step a
        #        second step).
        if len(known_param_indices):
            self.known_param_indices = known_param_indices
        else:
            self.known_param_indices = None

        # Allow MonotonicWithParams layers to be used on first net if needed.
        #  This counts all estimates except the first (num_y - 1) plus the
        #  contrast (1) plus all known params as "not needing to be monotone"
        layer_kwargs = [
            {'num_params': (num_y-1) + 1 + len(known_param_indices)},
            {},
        ]

        super().__init__(num_outputs_for_each_net=self.num_z,
                         layer_kwargs=layer_kwargs,
                         **network_setup_args)

    ###########################################################################
    #
    #  Methods overridden from _SimulatorNet
    #
    ###########################################################################

    @tf.function
    def simulate_training_data(
            self,
            n: int,
    ) -> Tuple[
            NetInputBlob,
            None,
    ]:

        return self.sample_ys_and_params(n), None

    @tf.function
    def get_validation_set(
            self
    ) -> Tuple[
            NetInputBlob,
            None,
    ]:

        return self.validation_set_fn(), None

    @tf.function
    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: None = None,
    ) -> ttf.float32:

        outputs, jacobians, dz0_dcontrast = net_outputs
        neg_log_likelihoods = self.neg_log_likelihoods(outputs, jacobians)
        dz0_dcontrast_is_neg = 100. * tf.keras.activations.relu(dz0_dcontrast)
        loss = tf.math.reduce_mean(neg_log_likelihoods + dz0_dcontrast_is_neg)

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
    def num_z_total(self) -> int:
        return sum(self.num_z)

    @tf.function
    def net_inputs_from_contrast(
            self,
            contrast: Tensor1[tf32, Samples],
            ys: Tensor2[tf32, Samples, Ys],
            params: Tensor2[tf32, Samples, Params],
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], ...]:

        coords = self.coords_fn(ys)
        contrast = contrast[:, None]
        if self.known_param_indices is not None:
            known_params = tf.gather(params, self.known_param_indices, axis=1)
            contrast_net_inputs = tf.concat([coords, contrast, known_params],
                                            axis=1)
        else:
            contrast_net_inputs = tf.concat([coords, contrast], axis=1)

        other_net_inputs = tf.concat([coords, params], axis=1)
        return contrast_net_inputs, other_net_inputs

    @tf.function
    def net_inputs(
            self,
            input_blob: NetInputBlob,
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], ...]:

        # TODO: relate this to the contrast rather than "known param" naming
        ys, params = input_blob
        contrast = self.contrast_fn(params)
        return self.net_inputs_from_contrast(contrast, ys, params)

    @tf.function
    def neg_log_likelihoods(
            self,
            outputs: Tensor2[tf32, Samples, Zs],
            sample_jacobdets: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        eps = common.SMALLEST_LOGABLE_NUMBER

        jacobdets_floored = _utils._soft_floor_at_zero(sample_jacobdets)

        normal_pd = tfp.distributions.Normal(0.0, 1.0).prob(outputs)
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
    ]:

        ys, params = input_blob

        with tf.GradientTape(persistent=True) as tape:                         # type: ignore
            tape.watch(ys)
            contrast = self.contrast_fn(params)
            tape.watch(contrast)

            inputs = self.net_inputs_from_contrast(contrast, ys, params)
            zs = self._call_tf(inputs, training=training)
            z0 = zs[:, 0]

        dz0_dcontrast = tape.gradient(z0, contrast)
        jacobians = tape.batch_jacobian(zs, ys)
        jacobdets = tf.linalg.det(jacobians)
        del tape

        return zs, jacobdets, dz0_dcontrast                                    # type: ignore

    @tf.function
    def sample_params(
            self,
            n: int,
    ) -> Tensor2[tf32, Samples, Params]:

        return self.param_sampling_fn(n)

    @tf.function
    def sample_ys_and_params(
            self,
            n: int,
    ) -> NetInputBlob:

        params = self.sample_params(n)
        y = self.sampling_distribution_fn(params)
        return y, params
