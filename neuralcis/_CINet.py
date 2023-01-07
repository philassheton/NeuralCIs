import tensorflow as tf

from neuralcis._SinglePNet import _SinglePNet
from neuralcis._SimulatorNet import _SimulatorNet
import neuralcis.common as common

# typing imports
from typing import Callable, Tuple
from tensor_annotations.tensorflow import Tensor1, Tensor2
from neuralcis.common import Samples, Params, Estimates, NetInputs, NetOutputs
from neuralcis.common import KnownParams, NetTargetBlob
import tensor_annotations.tensorflow as ttf
tf32 = ttf.float32


class _CINet:
    def __init__(
            self,
            pnet: _SinglePNet,                # TODO: make a protocol for PNets
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor2[tf32, Samples, Estimates]
            ],
            num_known_param: int
    ) -> None:

        self.pnet = pnet
        self.sampling_distribution_fn = sampling_distribution_fn
        self.num_known_param = num_known_param

        self.validation_set = self.simulate_training_data(
            common.VALIDATION_SET_SIZE
        )

        self.cinet = _SimulatorNet(
            self.simulate_training_data,
            self.get_validation_set,
            self.loss,
            num_outputs=2
        )

    def fit(
            self,
            *args,
            precompute_optimum_loss: bool = True,
            **kwargs
    ) -> None:

        if precompute_optimum_loss:
            self.precompute_optimum_loss()
        self.cinet.fit(*args, **kwargs)

    def precompute_optimum_loss(self) -> None:
        self.cinet.set_validation_optimum_loss(tf.constant(0.))

    @tf.function
    def get_num_params(self) -> int:
        """Get the total number of parameters.

        At present, this only supports the SinglePNet so there must be
        precisely one unknown parameter plus the known parameters.
        """
        return self.num_known_param + 1

    @tf.function
    def sample_params(self, n: ttf.int32) -> Tensor2[tf32, Samples, Params]:
        return tf.random.uniform(
            (n, self.get_num_params()),
            tf.constant(common.PARAMS_MIN),
            tf.constant(common.PARAMS_MAX)
        )

    @tf.function
    def simulate_training_data(
            self,
            n: ttf.int32
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], NetTargetBlob]:

        params = self.sample_params(n)
        estimates = self.sampling_distribution_fn(params)
        target_p = tf.random.uniform((n,), tf.constant(0.), tf.constant(1.))

        known_params = self.known_params(params)
        inputs = self.net_inputs(estimates, known_params, target_p)
        outputs = (estimates, params, target_p)

        return inputs, outputs                                                 # type: ignore

    @tf.function
    def net_inputs(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            known_params: Tensor2[tf32, Samples, KnownParams],
            target_p: Tensor1[tf32, Samples]
    ) -> Tensor2[tf32, Samples, NetInputs]:

        return tf.concat([
            estimates, known_params, tf.transpose([target_p])
        ], axis=1)

    @tf.function
    def known_params(
            self,
            params: Tensor2[tf32, Samples, Params]
    ) -> Tensor2[tf32, Samples, KnownParams]:

        return params[:, 1:]                                                   # type: ignore

    @tf.function
    def plugin_first_param(
            self,
            params: Tensor2[tf32, Samples, Params],
            first_param: Tensor1[tf32, Samples]
    ):

        known_params = self.known_params(params)
        combined_params = tf.concat([
            tf.transpose([first_param]),
            known_params
        ], axis=1)
        return combined_params

    @tf.function
    def get_validation_set(
            self
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], NetTargetBlob]:

        return self.validation_set

    @tf.function
    def loss(
            self,
            net_outputs: Tensor2[tf32, Samples, NetOutputs],
            target_outputs: Tuple[
                Tensor2[tf32, Samples, Estimates],
                Tensor2[tf32, Samples, Params],
                Tensor1[tf32, Samples]
            ]
    ) -> ttf.float32:

        estimates, params, p = target_outputs

        lower, upper = self.output_activation(net_outputs, estimates)

        p_lower = self.run_pnet_plugin_first_param(estimates, params, lower)
        p_upper = self.run_pnet_plugin_first_param(estimates, params, upper)

        squared_errors = (tf.math.square(p_lower - p) +
                          tf.math.square(p_upper - p))

        return tf.reduce_mean(squared_errors)

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
    def ci(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            known_params: Tensor2[tf32, Samples, KnownParams],
            conf_levels: Tensor1[tf32, Samples]
    ):

        net_inputs = self.net_inputs(estimates, known_params, conf_levels)
        net_outputs = self.cinet.call_tf(net_inputs)

        lower, upper = self.output_activation(net_outputs, estimates)

        return lower, upper
