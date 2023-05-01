from neuralcis._simulator_net import _SimulatorNet

import neuralcis.common as common

import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore

from typing import Callable, Tuple
from neuralcis.common import Params, Ys, Samples, NetInputs
import tensor_annotations.tensorflow as ttf
from tensor_annotations.tensorflow import Tensor1, Tensor2
tf32 = ttf.float32


NetOutputBlob = Tuple[Tensor2[tf32, Samples, Ys],      # net outputs (z values)
                      Tensor1[tf32, Samples]]          # Jacobian determinants


class _ZNet(_SimulatorNet):
    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],                 # params
                Tensor2[tf32, Samples, Ys]                        # -> ys
            ],
            param_sampling_fn: Callable[
                [ttf.int32],                                      # n
                Tensor2[tf32, Samples, Params]                    # -> params
            ],
            validation_set_fn: Callable[
                [],
                Tuple[
                    Tensor2[tf32, Samples, Ys],                   # -> ys
                    Tensor2[tf32, Samples, Params]                # -> params
                ]],
            filename: str = "",
            **network_setup_args
    ) -> None:

        self.sampling_distribution_fn = sampling_distribution_fn
        self.validation_set_fn = validation_set_fn
        self.param_sampling_fn = param_sampling_fn

        ys, params = self.validation_set_fn()

        self.num_z = tf.shape(ys)[1]

        super().__init__(num_outputs=self.num_z,
                         filename=filename,
                         **network_setup_args)

    ###########################################################################
    #
    #  Methods overridden from _SimulatorNet
    #
    ###########################################################################

    @tf.function
    def simulate_training_data(
            self,
            n: ttf.int32
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], None]:

        ys, params = self.sample_ys_and_params(n)
        net_inputs = self.net_inputs(ys, params)
        target_outputs = None

        return net_inputs, target_outputs

    @tf.function
    def get_validation_set(
            self
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], None]:

        return self.net_inputs(*self.validation_set_fn()), None

    @tf.function
    def loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: None = None
    ) -> ttf.float32:

        outputs, jacobians_floored = net_outputs
        neg_log_likelihoods = self.neg_log_likelihoods(outputs,
                                                       jacobians_floored)
        loss = tf.math.reduce_sum(neg_log_likelihoods)

        tf.debugging.check_numerics(loss,
                                    "Na or inf in loss in multiple Z Net opt")

        return loss

    @tf.function
    def run_net_during_training(
            self,
            net: tf.keras.Model,
            net_inputs: Tensor2[tf32, Samples, NetInputs],
            training: ttf.bool = tf.constant(True)
    ) -> NetOutputBlob:

        return self.net_outputs_and_transformation_jacobdets(
            net=net,
            ys_and_params=net_inputs,
            training=training
        )

    @tf.function
    def call_tf(
            self,
            y: Tensor2[tf32, Samples, Ys],
            params: Tensor2[tf32, Samples, Params]
    ) -> Tensor2[tf32, Samples, Ys]:

        net_inputs = self.net_inputs(y, params)
        net_outputs = super().call_tf(net_inputs)
        return net_outputs                                                     # type: ignore

    def compute_optimum_loss(self) -> ttf.float32:
        # TODO: the individual losses here are not currently saved.  Need to
        #       rewrite _DataSaver to have functions that can be overridden
        #       instead of taking constructor args which need screwing with.

        # TODO: implement optimum loss for multi-znet.

        return tf.constant(0.)

    ###########################################################################
    #
    #  Non-Tensorflow members
    #
    ###########################################################################

    def __call__(self, *y_and_params) -> Tensor2[tf32, Samples, Ys]:
        y_and_params_parsed = common.combine_input_args_into_tensor(
            *y_and_params
        )
        y, params = self.separate_net_inputs(y_and_params_parsed)
        return self.call_tf(y, params)

    ###########################################################################
    #
    #  Tensorflow members
    #
    ###########################################################################

    @tf.function
    def net_inputs(
            self,
            ys: Tensor2[tf32, Samples, Ys],
            params: Tensor2[tf32, Samples, Params]
    ) -> Tensor2[tf32, Samples, NetInputs]:

        return tf.concat([ys, params], axis=1)

    @tf.function
    def separate_net_inputs(
            self,
            net_inputs: Tensor2[tf32, Samples, NetInputs]
    ) -> Tuple[
        Tensor2[tf32, Samples, Ys],
        Tensor2[tf32, Samples, Params]
    ]:

        y: Tensor2[tf32, Samples, Ys] = net_inputs[:, 0:self.num_z]            # type: ignore
        params: Tensor2[tf32, Samples, Params] = net_inputs[:, self.num_z:]    # type: ignore

        return y, params

    # work first additively in log space to avoid overflows
    @tf.function
    def neg_log_likelihoods(
            self,
            outputs: Tensor2[tf32, Samples, Ys],
            sample_jacobdets: Tensor1[tf32, Samples]
    ) -> Tensor1[tf32, Samples]:

        normal_pd = tfp.distributions.Normal(0.0, 1.0).prob(outputs)
        normal_pd_joint = tf.math.reduce_prod(normal_pd, axis=1) + 1e-37
        neg_log_likelihoods = (-tf.math.log(tf.transpose(normal_pd_joint))
                               - tf.math.log(sample_jacobdets + 1e-37))

        return neg_log_likelihoods

    @tf.function
    def net_outputs_and_transformation_jacobdets(
            self,
            net: tf.keras.Model,
            ys_and_params: Tensor2[tf32, Samples, NetInputs],
            training: bool
    ) -> Tuple[Tensor2[tf32, Samples, Ys], Tensor1[tf32, Samples]]:

        ys, params = self.separate_net_inputs(ys_and_params)

        with tf.GradientTape() as tape:                                        # type: ignore
            tape.watch(ys)
            net_inputs = self.net_inputs(ys, params)
            outputs = net(net_inputs, training)
        jacobians = tape.batch_jacobian(outputs, ys)
        jacobdets = tf.linalg.det(jacobians)

        # TODO: Consider whether this needs to adapt the scale of the variables
        #       used (I think not, now that all variables are in [-1, 1].)
        punitive_but_not_zero_jacobdets = 1e-10 * tf.math.sigmoid(jacobdets)
        jacobdets_floored = tf.math.maximum(
            jacobdets,
            punitive_but_not_zero_jacobdets
        )

        jacobian_diags = tf.linalg.diag_part(jacobians)
        punitive_but_not_zero_diags = 1e-10 * tf.math.sigmoid(jacobian_diags)
        jacobian_diags_floored = tf.math.maximum(
            jacobian_diags,
            punitive_but_not_zero_diags
        )

        # TODO: check whether this step is really necessary (or helpful?)
        jacobdets_punished = tf.math.minimum(
            jacobdets_floored,
            tf.math.reduce_prod(jacobian_diags_floored, axis=1)
        )

        return outputs, jacobdets_punished

    @tf.function
    def sample_params(
            self,
            n: ttf.int32
    ) -> Tensor2[tf32, Samples, Params]:

        return self.param_sampling_fn(n)

    @tf.function
    def sample_ys_and_params(
            self,
            n: ttf.int32
    ) -> Tuple[
        Tensor2[tf32, Samples, Ys],
        Tensor2[tf32, Samples, Params]
    ]:

        params = self.sample_params(n)
        y = self.sampling_distribution_fn(params)
        return y, params
