import tensorflow as tf

from neuralcis._p_net import _PNet
from neuralcis._simulator_net import _SimulatorNet
import neuralcis.common as common

# typing imports
from typing import Callable, Tuple, List, Sequence
from tensor_annotations.tensorflow import Tensor1, Tensor2
from neuralcis.common import Samples, Params, Estimates, NetInputs, NetOutputs
from neuralcis.common import KnownParams
import tensor_annotations.tensorflow as ttf
tf32 = ttf.float32


NetInputBlob = Tuple[Tensor2[tf32, Samples, Estimates],
                     Tensor2[tf32, Samples, KnownParams],
                     Tensor1[tf32, Samples]]                         # p-values
NetOutputBlob = Tensor2[tf32, Samples, NetOutputs]
NetTargetBlob = Tuple[Tensor2[tf32, Samples, Estimates],
                      Tensor2[tf32, Samples, Params],
                      Tensor1[tf32, Samples]]                        # p-values


class _CINet(_SimulatorNet):
    relative_loss_increase_tol = common.REL_LOSS_INCREASE_TOL_CI_NET
    def __init__(
            self,
            pnet: _PNet,               # TODO: make a protocol for PNets
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor2[tf32, Samples, Estimates]
            ],
            sample_params_fn: Callable[
                [int],
                Tensor2[tf32, Samples, Params],
            ],
            num_param: int,
            known_param_indices: Sequence[int],
            **network_setup_args,
    ) -> None:

        _SimulatorNet.__init__(self,
                               num_inputs_for_each_net=[num_param + 1],
                               num_outputs_for_each_net=[2],
                               **network_setup_args)

        self.sample_params = sample_params_fn
        self.pnet = pnet
        self.sampling_distribution_fn = sampling_distribution_fn
        self.num_param = num_param
        self.known_param_indices = known_param_indices

    ###########################################################################
    #
    #  Methods overridden from _SimulatorNet
    #
    ###########################################################################

    @tf.function
    def simulate_training_data(
            self,
    ) -> Tuple[NetInputBlob, NetTargetBlob]:

        n = self.batch_size
        params = self.sample_params(n)
        estimates = self.sampling_distribution_fn(params)
        target_p = tf.random.uniform((n,), tf.constant(0.), tf.constant(1.))

        known_params = self.known_params(params)
        inputs = (estimates, known_params, target_p)
        outputs = (estimates, params, target_p)

        return inputs, outputs

    @tf.function
    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: NetTargetBlob,
    ) -> ttf.float32:

        estimates, params, p = target_outputs

        lower, upper = self.output_activation(net_outputs, estimates)

        p_lower = self.p_from_pnet(estimates, lower, params)
        p_upper = self.p_from_pnet(estimates, upper, params)

        squared_errors = (tf.math.square(p_lower - p) +
                          tf.math.square(p_upper - p))

        return tf.reduce_mean(squared_errors)                                  # type: ignore

    @tf.function
    def compute_optimum_loss(self) -> ttf.float32:
        return tf.constant(0.)

    ###########################################################################
    #
    #  Tensorflow members
    #
    ###########################################################################

    @tf.function
    def net_inputs(
            self,
            net_input_blob: NetInputBlob,
    ) -> List[Tensor2[tf32, Samples, NetInputs]]:

        estimates, known_params, target_p = net_input_blob
        ins = tf.concat([
            estimates, known_params, tf.transpose([target_p])
        ], axis=1)

        return [ins]

    @tf.function
    def known_params(
            self,
            params: Tensor2[tf32, Samples, Params],
    ) -> Tensor2[tf32, Samples, KnownParams]:

        return tf.gather(params, self.known_param_indices, axis=1)

    @tf.function
    def p_from_pnet(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            contrast: Tensor1[tf32, Samples],
            params: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        known_params = self.known_params(params)
        return self.pnet.p_from_contrast(estimates, contrast, known_params)

    @tf.function
    def output_activation(
            self,
            net_outputs: Tensor2[tf32, Samples, NetOutputs],
            estimates: Tensor2[tf32, Samples, Estimates],
    ) -> Tuple[
        Tensor1[tf32, Samples],
        Tensor1[tf32, Samples],
    ]:

        lower = estimates[:, 0] - tf.math.exp(net_outputs[:, 0])
        upper = estimates[:, 0] + tf.math.exp(net_outputs[:, 1])

        return lower, upper                                                    # type: ignore

    @tf.function
    def ci(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            known_params: Tensor2[tf32, Samples, KnownParams],
            conf_levels: Tensor1[tf32, Samples],
    ) -> Tuple[Tensor2[tf32, Samples, Params], Tensor2[tf32, Samples, Params]]:

        net_outputs = self.call_tf((estimates, known_params, conf_levels))
        lower, upper = self.output_activation(net_outputs, estimates)

        return lower, upper
