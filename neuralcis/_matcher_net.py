import tensorflow as tf

from neuralcis._simulator_net import _SimulatorNet, tf32
from neuralcis import common
from typing import Sequence, Tuple, Optional

from typing import Callable
from tensor_annotations import tensorflow as ttf
from tensor_annotations.tensorflow import Tensor2
from neuralcis.common import Samples, NetInputs, NetOutputs
from neuralcis.common import NetInputBlob, NetOutputBlob

NetTargetBlob = Tensor2[tf32, Samples, NetOutputs]


class _MatcherNet(_SimulatorNet):

    """The goal of a matcher net is to take a complex blobs of other nets
    and train a single network to mimic their outputs.  At present this is
    used to initialise the _ZNet from the _ZNetConstrained, but could also
    be used for any other situation where the setup is getting to complex
    and we need to simplify things a bit."""

    def __init__(
            self,
            function_to_match: Callable[
                [NetInputBlob],
                Tensor2[tf32, Samples, NetOutputs]
            ],
            inputs_sampling_fn: Callable[
                [int],
                NetInputBlob
            ],
            internal_nets: Sequence[_SimulatorNet],
            internal_net_inputs: Callable[
                [NetInputBlob],
                Sequence[Tensor2[tf32, Samples, NetInputs]]
            ],
    ) -> None:

        self.function_to_match = function_to_match
        self.inputs_sampling_fn = inputs_sampling_fn
        self.internal_net_inputs = internal_net_inputs
        super().__init__(
            internal_nets=internal_nets,
        )
        n_validation = common.VALIDATION_SET_SIZE
        validation_input_blob = self.inputs_sampling_fn(n_validation)
        self.validation_set = self.inputs_outputs_from_input_blob(
            validation_input_blob
        )

    @tf.function
    def simulate_training_data(
            self, n: int,
    ) -> Tuple[
        NetInputBlob,
        NetTargetBlob,
    ]:
        input_blob = self.inputs_sampling_fn(n)
        return self.inputs_outputs_from_input_blob(input_blob)

    @tf.function
    def inputs_outputs_from_input_blob(
            self,
            input_blob: NetInputBlob,
    ) -> Tuple[
        Sequence[Tensor2[tf32, Samples, NetInputs]],
        Tensor2[tf32, Samples, NetOutputs]
    ]:

        net_inputs = self.internal_net_inputs(input_blob)
        outputs = self.function_to_match(input_blob)
        return net_inputs, outputs

    @tf.function
    def get_validation_set(
            self
    ) -> Tuple[
        NetInputBlob,
        NetTargetBlob,
    ]:

        return self.validation_set

    @tf.function
    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: Optional[NetTargetBlob] = None,
    ) -> ttf.float32:

        errors = net_outputs - target_outputs
        return tf.math.sqrt(tf.reduce_mean(tf.square(errors)))
