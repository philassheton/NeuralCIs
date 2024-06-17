from neuralcis._simulator_net import _SimulatorNet
from neuralcis._layers import _DefaultIn, _DefaultHid, _DefaultOut
from neuralcis._layers import _MonotonicLayer
from neuralcis import common

import tensorflow as tf

# Typing
from typing import Tuple, Sequence, List, Optional, Callable
from neuralcis.common import Samples, Params, Estimates, NetInputs
from neuralcis.common import NetOutputs, NetTargetBlob, NetOutputBlob
from tensor_annotations import tensorflow as ttf
from tensor_annotations.tensorflow import Tensor2
tf32 = ttf.float32


class _CoordiNet(_SimulatorNet):
    def __init__(
            self,
            estimatornet: _SimulatorNet,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor2[tf32, Samples, Estimates]
            ],
            param_sampling_fn: Callable[
                [int],                                                         # n
                Tensor2[tf32, Samples, Params]                                 # -> params
            ],
            filename: str = "",
            **network_setup_args
    ):

        # TODO: Rethink how all this ties together as currently rather
        #       fragile.  E.g. here have to initialise the Model before
        #       assigning our side-nets.  But then we have to wait until
        #       they are assigned to initialise the simulatornet.
        tf.keras.Model.__init__(self)
        self.estimatornet = estimatornet
        self.sampling_distribution_fn = sampling_distribution_fn
        self.param_sampling_fn = param_sampling_fn
        self.validation_ys, _ = self.simulate_training_data(
            common.VALIDATION_SET_SIZE
        )
        self.num_y = self.validation_ys[0].shape[1]

        _SimulatorNet.__init__(
            self,
            (1, self.num_y - 1),
            first_layer_type_or_types=(_MonotonicLayer, _DefaultIn),
            hidden_layer_type_or_types=(_MonotonicLayer, _DefaultHid),
            output_layer_type_or_types=(_MonotonicLayer, _DefaultOut),
            filename=filename,
            **network_setup_args
        )

    @tf.function
    def simulate_training_data(
            self,
            n: int,
    ) -> Tuple[
        List[Tensor2[tf32, Samples, NetInputs]],
        Optional[NetTargetBlob]
    ]:

        params = self.param_sampling_fn(n)
        estimates = self.sampling_distribution_fn(params)
        inputs: Tensor2[tf32, Samples, NetInputs] = estimates                  # type: ignore

        return [inputs], None

    @tf.function
    def get_validation_set(
            self
    ) -> Tuple[Sequence[Tensor2[tf32, Samples, NetInputs]],
               Optional[NetTargetBlob]]:

        return self.validation_ys, None

    @tf.function
    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: Optional[NetTargetBlob] = None
    ) -> ttf.float32:

        coords, jacobians = net_outputs

        # Volume-preserving map
        jacobdets = tf.linalg.det(jacobians)
        eps = 1e-37
        loss = tf.reduce_mean(tf.abs(tf.math.log(tf.maximum(jacobdets, eps))))

        return loss

    @tf.function
    def call_tf_training(
            self,
            net_ins: Sequence[Tensor2[tf32, Samples, NetInputs]]
    ) -> NetOutputBlob:

        with tf.GradientTape() as tape:
            tape.watch(net_ins[0])
            coords = self._call_tf(net_ins, training=True)
        jacobians = tape.batch_jacobian(coords, net_ins[0])

        return coords, jacobians

    @tf.function
    def _call_tf(
            self,
            net_ins: Sequence[Tensor2[tf32, Samples, NetInputs]],
            training: bool
    ) -> Tensor2[tf32, Samples, NetOutputs]:

        estimate = self.estimatornet._call_tf(net_ins, training=training)
        coords = super()._call_tf([estimate, net_ins[0]], training=training)

        return coords
