import tensorflow as tf

from typing import List
import tensor_annotations.tensorflow as ttf
from tensor_annotations.tensorflow import Tensor2
from neuralcis.common import Samples, LayerInputs, LayerOutputs
tf32 = ttf.float32


class _FiftyFiftyLayer(tf.keras.layers.Layer):
    def __init__(self, n_outputs_per_activation: int) -> None:
        super(_FiftyFiftyLayer, self).__init__()
        self.n_outputs_per_activation = n_outputs_per_activation
        self.n_outputs = n_outputs_per_activation * 2

    def build(self, input_shape: List) -> None:
        n_inputs = int(input_shape[-1])

        self.kernel = self.add_weight(
            "kernel",
            shape=(n_inputs, self.n_outputs),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        self.bias = self.add_weight(
            "bias",
            shape=(self.n_outputs, ),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.1,
                                                            maxval=0.1),
            trainable=True
        )

    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs]
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        potentials = tf.linalg.matmul(inputs, self.kernel) + self.bias
        tanh_outputs = potentials[:, 0:self.n_outputs_per_activation]
        elu_outputs = potentials[:, self.n_outputs_per_activation:]
        activations = tf.concat([
            tf.keras.activations.tanh(tanh_outputs),
            tf.keras.activations.elu(elu_outputs)
        ], axis=1)

        return activations
