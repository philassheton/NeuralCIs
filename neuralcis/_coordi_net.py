from neuralcis._simulator_net import _SimulatorNet
from neuralcis._layers import _DefaultIn, _DefaultHid, _DefaultOut
from neuralcis._layers import _MonotonicTanhLayer, _MonotonicLinearLayer
from neuralcis import common

import tensorflow as tf

# Typing
from typing import Tuple, Sequence, List, Optional, Callable
from neuralcis.common import Samples, Params, Estimates, NetInputs
from neuralcis.common import NetOutputs, NetTargetBlob, NetOutputBlob
from tensor_annotations import tensorflow as ttf
from tensor_annotations.tensorflow import Tensor2
tf32 = ttf.float32


NetInputBlob = Tensor2[tf32, Samples, Estimates]


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
        self.num_y = self.validation_ys.shape[1]

        _SimulatorNet.__init__(
            self,
            (1, self.num_y - 1),
            first_layer_type_or_types=(_MonotonicTanhLayer, _DefaultIn),
            hidden_layer_type_or_types=(_MonotonicTanhLayer, _DefaultHid),
            output_layer_type_or_types=(_MonotonicLinearLayer, _DefaultOut),
            filename=filename,
            **network_setup_args
        )

    @tf.function
    def simulate_training_data(
            self,
            n: int,
    ) -> Tuple[
        NetInputBlob,
        None,
    ]:

        params = self.param_sampling_fn(n)
        estimates = self.sampling_distribution_fn(params)

        return estimates, None                                                 # type: ignore

    @tf.function
    def get_validation_set(
            self,
    ) -> Tuple[NetInputBlob, None]:

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
        punitive_but_not_zero_jacobdets = 1e-10 * tf.math.sigmoid(jacobdets)
        jacobdets_floored = tf.math.maximum(
            jacobdets,
            punitive_but_not_zero_jacobdets,
        )
        loss = tf.reduce_mean(tf.abs(tf.math.log(jacobdets_floored)))

        return loss

    @tf.function
    def call_tf_training(
            self,
            estimates: NetInputBlob,
    ) -> NetOutputBlob:

        with tf.GradientTape() as tape:
            tape.watch(estimates)
            net_ins = self.net_inputs(estimates)
            coords = self._call_tf(net_ins, training=True)
        jacobians = tape.batch_jacobian(coords, estimates)

        return coords, jacobians

    @tf.function
    def net_inputs(
            self,
            estimates: NetInputBlob,
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs]]:

        con_estimate = self.estimatornet.call_tf(estimates)
        return [con_estimate, estimates]                                       # Type: ignore

