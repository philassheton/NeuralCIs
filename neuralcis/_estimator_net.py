import tensorflow as tf

from neuralcis._simulator_net import _SimulatorNet
from neuralcis import common

# Typing
from typing import Tuple, Optional, Callable
from neuralcis.common import Samples, Params, Estimates
from neuralcis.common import NetInputs, NetOutputBlob
from tensor_annotations import tensorflow as ttf
from tensor_annotations.tensorflow import Tensor1, Tensor2

tf32 = ttf.float32
NetInputBlob = Tensor2[tf32, Samples, Estimates]
NetTargetBlob = Tensor1[tf32, Samples]


class _EstimatorNet(_SimulatorNet):
    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],                              # params
                Tensor2[tf32, Samples, Estimates]                              # -> ys
            ],
            param_sampling_fn: Callable[
                [int],                                                         # n
                Tensor2[tf32, Samples, Params]                                 # -> params
            ],
            contrast_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor1[tf32, Samples]
            ],
            num_samples_per_params: int = 1,                                   # This can be adjusted later if we want to explore methods that take multiple samples from the same params combination
            filename="",
            **network_setup_args,
    ):

        self.num_samples_per_params = num_samples_per_params

        self.sampling_distribution_fn = sampling_distribution_fn
        self.param_sampling_fn = param_sampling_fn
        self.contrast_fn = contrast_fn
        self.validation_ys, self.validation_targets = \
            self.simulate_training_data(common.VALIDATION_SET_SIZE)

        super().__init__([1], filename=filename, **network_setup_args)

    @tf.function
    def simulate_training_data(
            self,
            num_param_samples: int,
    ) -> Tuple[
        NetInputBlob,
        NetTargetBlob,
    ]:

        params = self.param_sampling_fn(num_param_samples)
        params_replicated = tf.tile(params, (self.num_samples_per_params, 1))
        y_samples = self.sampling_distribution_fn(params_replicated)
        targets = self.contrast_fn(params)

        return y_samples, targets

    @tf.function
    def get_validation_set(
            self,
    ) -> Tuple[
        NetInputBlob,
        NetTargetBlob,
    ]:

        return self.validation_ys, self.validation_targets

    @tf.function
    def net_inputs(
            self,
            estimates: NetInputBlob,
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], ...]:

        return (estimates, )                                                   # Type: ignore

    @tf.function
    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: Optional[NetTargetBlob] = None,
    ) -> ttf.float32:

        c = target_outputs[:, None]

        # TODO: if we decide to abandon multiple samples per params, then this
        #       should all be collapsed.  Leaving it here for now as not quite
        #       clear yet on the best route here.
        c_hat = tf.stack(tf.split(net_outputs,
                                  self.num_samples_per_params,
                                  axis=0),
                         axis=2)

        c_hat_means = tf.math.reduce_mean(c_hat, axis=2)
        losses = tf.math.square(c - c_hat_means)
        loss = tf.math.reduce_mean(losses)

        return loss
