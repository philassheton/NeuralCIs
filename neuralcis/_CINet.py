import tensorflow as tf

from neuralcis._SinglePNet import _SinglePNet
from neuralcis._SimulatorNet import _SimulatorNet
import neuralcis.common as common

# typing imports
from typing import Callable, Tuple
from tensor_annotations.tensorflow import Tensor1, Tensor2
from neuralcis.common import Samples, Params, Estimates, NetInputs, NetOutputs
from neuralcis.common import NetTargetBlob
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

    def fit(self, *args):
        self.cinet.fit(*args)

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

        inputs = self.net_inputs(estimates, params, target_p)
        outputs = (estimates, params, target_p)

        return inputs, outputs                                                 # type: ignore

    @tf.function
    def net_inputs(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            params: Tensor2[tf32, Samples, Params],
            target_p: Tensor1[tf32, Samples]
    ) -> Tensor2[tf32, Samples, NetInputs]:

        known_params = self.known_params(params)
        return tf.concat([
            estimates, known_params, tf.transpose([target_p])
        ], axis=1)

    @tf.function
    def known_params(
            self,
            params: Tensor2[tf32, Samples, Params]
    ) -> Tensor2[tf32, Samples, Params]:

        return params[:, 1:]

    @tf.function
    def pnet_inputs(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            params: Tensor2[tf32, Samples, Params],
    ) -> Tensor2[tf32, Samples, NetInputs]:

        return tf.concat([estimates, params], axis=1)

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

        lower = estimates[:, 0] - tf.math.exp(net_outputs[:, 0])
        upper = estimates[:, 0] + tf.math.exp(net_outputs[:, 1])

        pnet_params_lower = self.plugin_first_param(params, lower)
        pnet_params_upper = self.plugin_first_param(params, upper)

        p_lower = self.pnet.p(estimates, pnet_params_lower)
        p_upper = self.pnet.p(estimates, pnet_params_upper)

        squared_errors = (tf.math.square(p_lower - p) +
                          tf.math.square(p_upper - p))

        return tf.reduce_sum(squared_errors)
