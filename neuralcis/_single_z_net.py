from neuralcis._simulator_net import _SimulatorNet

import neuralcis.common as common

import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
import numpy as np

from typing import Callable, Tuple, Any
from neuralcis.common import Params, Samples, NetInputs, NumApproximations
import tensor_annotations.tensorflow as ttf
from tensor_annotations.tensorflow import Tensor1, Tensor2
tf32 = ttf.float32


NetOutputBlob = Tuple[Tensor1[tf32, Samples],          # net outputs (z values)
                      Tensor1[tf32, Samples]]      # dz/dy derivs wrt focal var


# estimates the density of y given params
class _SingleZNet(_SimulatorNet):
    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],                 # params
                Tensor1[tf32, Samples]                            # y
            ],
            param_sampling_fn: Callable[
                [ttf.int32],                                      # n
                Tensor2[tf32, Samples, Params]                    # params
            ],
            validation_set_fn: Callable[
                [],
                Tuple[
                    Tensor1[tf32, Samples],                       # y
                    Tensor2[tf32, Samples, Params]                # params
                ]],
            filename: str = "",
            **network_setup_args
    ) -> None:

        self.sampling_distribution_fn = sampling_distribution_fn
        self.validation_set_fn = validation_set_fn
        self.param_sampling_fn = param_sampling_fn

        y, params = self.validation_set_fn()
        self.validation_optimum_losses = tf.Variable(y * 0.)

        _SimulatorNet.__init__(self, filename=filename, **network_setup_args)

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

        y, params = self.sample_y_and_params(n)
        net_inputs = self.net_inputs(y, params)
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

        outputs, sample_grads_floored = net_outputs
        neg_log_likelihoods = self.neg_log_likelihoods(outputs,
                                                       sample_grads_floored)
        loss = tf.math.reduce_sum(neg_log_likelihoods)

        tf.debugging.check_numerics(loss,
                                    "Na or inf in loss in single Z Net opt")

        return loss

    @tf.function
    def run_net_during_training(
            self,
            net: tf.keras.Model,
            net_inputs: Tensor2[tf32, Samples, NetInputs],
            training: ttf.bool = tf.constant(True)
    ) -> NetOutputBlob:

        return self.net_outputs_and_transformation_derivs(
            net=net,
            y_and_params=net_inputs,
            training=training
        )

    @tf.function
    def call_tf(
            self,
            y: Tensor1[tf32, Samples],
            params: Tensor2[tf32, Samples, Params]
    ) -> Tensor1[tf32, Samples]:

        net_inputs = self.net_inputs(y, params)
        net_outputs = super().call_tf(net_inputs)
        return self.pull_first_column(net_outputs)

    def compute_optimum_loss(self) -> ttf.float32:
        # TODO: the individual losses here are not currently saved.  Need to
        #       rewrite _DataSaver to have functions that can be overridden
        #       instead of taking constructor args which need screwing with.

        self.validation_optimum_losses.assign(
            self.estimate_perfect_error_for_validation_set()[:, -1]
        )
        return self.total_ideal_loss_accounting_for_missings()

    ###########################################################################
    #
    #  Non-Tensorflow members
    #
    ###########################################################################

    def __call__(self, y, *argv) -> Tensor1[tf32, Samples]:
        params = common.combine_input_args_into_tensor(*argv)
        return self.call_tf(y, params)

    ###########################################################################
    #
    #  Tensorflow members
    #
    ###########################################################################

    @tf.function
    def net_inputs(
            self,
            y: Tensor1[tf32, Samples],
            params: Tensor2[tf32, Samples, Params]
    ) -> Tensor2[tf32, Samples, NetInputs]:

        return tf.concat([
            tf.transpose([y]),
            params
        ], axis=1)

    # work first additively in log space to avoid overflows
    @tf.function
    def neg_log_likelihoods(
            self,
            outputs: Tensor1[tf32, Samples],
            sample_gradients_floored: Tensor1[tf32, Samples]
    ) -> Tensor1[tf32, Samples]:

        normal_pd = tfp.distributions.Normal(0.0, 1.0).prob(outputs) + 1e-37
        neg_log_likelihoods = (-tf.math.log(tf.transpose(normal_pd))
                               - tf.math.log(sample_gradients_floored + 1e-37))

        return neg_log_likelihoods

    @tf.function
    def net_outputs_and_transformation_derivs(
            self,
            net: tf.keras.Model,
            y_and_params: Tensor2[tf32, Samples, NetInputs],
            training: bool
    ) -> Tuple[Tensor1[tf32, Samples], Tensor1[tf32, Samples]]:

        y = self.pull_first_column(y_and_params)
        params = self.get_params_from_net_inputs(y_and_params)

        with tf.GradientTape() as tape:                                        # type: ignore
            tape.watch(y)
            net_inputs = self.net_inputs(y, params)
            outputs = net(net_inputs, training)[:, 0]
        sample_gradients = tape.gradient(outputs, y)

        # TODO: Consider whether this needs to adapt the scale of the variables
        #       used (I think not, now that all variables are in [-1, 1].)
        punitive_but_not_zero = 1e-10 * tf.math.sigmoid(sample_gradients)
        sample_gradients_floored = tf.math.maximum(sample_gradients,
                                                   punitive_but_not_zero)

        return outputs, sample_gradients_floored

    @tf.function
    def sample_params(
            self,
            n: ttf.int32
    ) -> Tensor2[tf32, Samples, Params]:

        return self.param_sampling_fn(n)

    @tf.function
    def sample_y_and_params(
            self,
            n: ttf.int32
    ) -> Tuple[Tensor1[tf32, Samples], Tensor2[tf32, Samples, Params]]:

        params = self.sample_params(n)
        y = self.sampling_distribution_fn(params)
        return y, params

    ###########################################################################
    #
    #  Estimation of ideal error
    #
    ###########################################################################

    # TODO: Is there a cleaner way than concatting y_and_params so we can map
    #  over them?  map allows you to do (y, params) but then it asks for self..

    @ tf.function
    def estimate_perfect_error_for_one_datapoint(
            self,
            y_and_params: Tensor1[tf32, NetInputs]
    ) -> Tensor1[tf32, NumApproximations]:

        num_samples = common.NUM_SAMPLES_FOR_IDEAL_ERROR_ESTIMATION
        largest_gap = common.GAP_BETWEEN_SAMPLES_FOR_PDF_ESTIMATION

        y = y_and_params[0]
        params = y_and_params[1:]

        params_repeated = tf.repeat([params], num_samples, axis=0)
        ys = self.sampling_distribution_fn(params_repeated)
        centred = ys - y

        centred_padded = tf.concat([
            centred,
            tf.repeat(np.inf, largest_gap),
            tf.repeat(-np.inf, largest_gap)
        ], axis=0)

        points_above = -tf.math.top_k(-centred_padded[centred_padded > 0],
                                      k=largest_gap).values
        points_below = tf.math.top_k(centred_padded[centred_padded <= 0],
                                     k=largest_gap).values
        gaps = points_above - points_below
        gap_widths = tf.cast(tf.range(largest_gap) * 2 + 1, tf.float32)
        pdf_estimates = gap_widths / (num_samples * gaps)

        return -tf.math.log(pdf_estimates)

    @tf.function
    def estimate_perfect_error_for_validation_set(
            self
    ) -> Tensor2[tf32, Samples, NumApproximations]:

        y, params = self.validation_set_fn()
        y_and_params = self.net_inputs(y, params)
        errors = tf.map_fn(self.estimate_perfect_error_for_one_datapoint,
                           y_and_params)
        return errors

    # TODO: The optimum error estimator still calculates a series of estimates
    #       and now only the last one is used, so rewrite the functions to only
    #       pass around one set of approximations.

    @tf.function
    def total_ideal_loss_accounting_for_missings(
            self
    ) -> ttf.float32:

        losses = self.validation_optimum_losses
        total_loss = tf.reduce_sum(losses[                                     # type: ignore
            tf.logical_not(tf.math.is_inf(losses))
        ])

        n_missing_losses = tf.reduce_sum(tf.cast(tf.math.is_inf(losses),
                                                 tf.float32))
        n_validation_samples = tf.cast(tf.size(losses), tf.float32)
        inflation_for_missing_values = n_validation_samples / (
                n_validation_samples - n_missing_losses
        )

        # allow max_value 10% missing values
        tf.debugging.assert_less_equal(
            inflation_for_missing_values,
            1. / (1. - common.MAX_PROPORTION_MISSING_VALUES_TO_TOLERATE)
        )

        return total_loss * inflation_for_missing_values

    ###########################################################################
    #
    #  Type conversions
    #
    ###########################################################################

    # TODO: The following functions only currently exist as a coping mechanism
    #  for gaps in the tensorflow_annotations (it does not seem to understand
    #  slicing correctly at present).  So these functions provide a controlled
    #  environment to ignore type errors.  Ideally fix up the stubs.
    @tf.function
    def pull_first_column(
            self,
            y_and_params: Tensor2[tf32, Samples, Any]
    ) -> Tensor1[tf32, Samples]:

        return y_and_params[:, 0]                                              # type: ignore

    @tf.function
    def get_params_from_net_inputs(
            self,
            y_and_params: Tensor2[tf32, Samples, NetInputs]
    ) -> Tensor2[tf32, Samples, Params]:

        return y_and_params[:, 1:]                                             # type:ignore
