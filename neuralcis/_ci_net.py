import tensorflow as tf

from neuralcis._p_net import _PNet
from neuralcis._simulator_net import _SimulatorNet
import neuralcis.common as common

# typing imports
from typing import Callable, Tuple, List
from tensor_annotations.tensorflow import Tensor1, Tensor2
from neuralcis.common import Samples, Params, Estimates, NetInputs, NetOutputs
from neuralcis.common import KnownParams
import tensor_annotations.tensorflow as ttf
tf32 = ttf.float32


NetOutputBlob = Tensor2[tf32, Samples, NetOutputs]
NetTargetBlob = Tuple[Tensor2[tf32, Samples, Estimates],
                      Tensor2[tf32, Samples, Params],
                      Tensor1[tf32, Samples]]                        # p-values


class _CINet(_SimulatorNet):
    def __init__(
            self,
            pnet: _PNet,               # TODO: make a protocol for PNets
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor2[tf32, Samples, Estimates]
            ],
            num_param: int,
            filename: str = "",
            **network_setup_args
    ) -> None:

        self.pnet = pnet
        self.sampling_distribution_fn = sampling_distribution_fn
        self.num_param = num_param

        self.validation_set = self.simulate_training_data(
            common.VALIDATION_SET_SIZE
        )

        _SimulatorNet.__init__(self,
                               num_outputs=[2],
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
    ) -> Tuple[List[Tensor2[tf32, Samples, NetInputs]], NetTargetBlob]:

        params = self.sample_params(n)
        estimates = self.sampling_distribution_fn(params)
        target_p = tf.random.uniform((n,), tf.constant(0.), tf.constant(1.))

        known_params = self.known_params(params)
        inputs = self.net_inputs(estimates, known_params, target_p)
        outputs = (estimates, params, target_p)

        return inputs, outputs

    @tf.function
    def get_validation_set(
            self
    ) -> Tuple[List[Tensor2[tf32, Samples, NetInputs]], NetTargetBlob]:

        return self.validation_set

    @tf.function
    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: NetTargetBlob
    ) -> ttf.float32:

        estimates, params, p = target_outputs

        lower, upper = self.output_activation(net_outputs, estimates)

        p_lower = self.run_pnet_plugin_first_param(estimates, params, lower)
        p_upper = self.run_pnet_plugin_first_param(estimates, params, upper)

        squared_errors = (tf.math.square(p_lower - p) +
                          tf.math.square(p_upper - p))

        return tf.reduce_mean(squared_errors)

    @tf.function
    def compute_optimum_loss(self) -> ttf.float32:
        return tf.constant(0.)

    ###########################################################################
    #
    #  Tensorflow members
    #
    ###########################################################################

    @tf.function
    def sample_params(self, n: ttf.int32) -> Tensor2[tf32, Samples, Params]:
        return tf.random.uniform(
            (n, self.num_param),
            tf.constant(common.PARAMS_MIN),
            tf.constant(common.PARAMS_MAX)
        )

    @tf.function
    def net_inputs(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            known_params: Tensor2[tf32, Samples, KnownParams],
            target_p: Tensor1[tf32, Samples]
    ) -> List[Tensor2[tf32, Samples, NetInputs]]:

        ins = tf.concat([
            estimates, known_params, tf.transpose([target_p])
        ], axis=1)

        return [ins]

    @tf.function
    def known_params(
            self,
            params: Tensor2[tf32, Samples, Params]
    ) -> Tensor2[tf32, Samples, KnownParams]:

        # TODO: The heart of the bit that will need to change to accommodate
        #       simultaneous CIs.  At present it is only taking the first
        #       param as unknown, and taking the other params from the null
        #       hypothesis.

        return params[:, 1:]                                                   # type: ignore

    @tf.function
    def plugin_first_param(
            self,
            params: Tensor2[tf32, Samples, Params],
            first_param: Tensor1[tf32, Samples]
    ):

        # TODO: Also will need to change to accommodate more parameters.

        known_params = self.known_params(params)
        combined_params = tf.concat([
            tf.transpose([first_param]),
            known_params
        ], axis=1)
        return combined_params

    @tf.function
    def run_pnet_plugin_first_param(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            params: Tensor2[tf32, Samples, Params],
            first_param: Tensor1[tf32, Samples]
    ) -> Tensor1[tf32, Samples]:

        plugin_params = self.plugin_first_param(params, first_param)
        return self.pnet.p(estimates, plugin_params)

    @tf.function
    def output_activation(
            self,
            net_outputs: Tensor2[tf32, Samples, NetOutputs],
            estimates: Tensor2[tf32, Samples, Estimates]
    ) -> Tuple[Tensor1[tf32, Samples], Tensor1[tf32, Samples]]:

        lower = estimates[:, 0] - tf.math.exp(net_outputs[:, 0])
        upper = estimates[:, 0] + tf.math.exp(net_outputs[:, 1])

        return lower, upper

    @tf.function
    def add_known_params(
            self,
            good_params: Tensor1[tf32, Samples],
            known_params: Tensor2[tf32, Samples, KnownParams]
    ) -> Tensor2[tf32, Samples, Params]:

        return tf.concat([tf.transpose([good_params]), known_params], axis=1)  # type: ignore

    @tf.function
    def ci(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            known_params: Tensor2[tf32, Samples, KnownParams],
            conf_levels: Tensor1[tf32, Samples]
    ) -> Tuple[Tensor2[tf32, Samples, Params], Tensor2[tf32, Samples, Params]]:

        net_inputs = self.net_inputs(estimates, known_params, conf_levels)
        net_outputs = self.call_tf(net_inputs)

        lower, upper = self.output_activation(net_outputs, estimates)

        # when dealing with transformed values, we need to know all params
        #    because other params might be used in the de-transform.
        # TODO: Make sure that the multidimensional case uses the right set of
        #       params for the de-transform.
        lower_full = self.add_known_params(lower, known_params)
        upper_full = self.add_known_params(upper, known_params)

        return lower_full, upper_full
