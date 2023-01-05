import tensorflow as tf

from typing import List
import tensor_annotations.tensorflow as ttf
from tensor_annotations.tensorflow import Tensor2
from neuralcis.common import Samples, LayerInputs, LayerOutputs
tf32 = ttf.float32


class FiftyFiftyLayer(tf.keras.layers.Layer):
    def __init__(self, n_outputs_per_activation: int) -> None:
        super(FiftyFiftyLayer, self).__init__()
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
            initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            trainable=True
        )

    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs]
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        potentials = tf.linalg.matmul(inputs, self.kernel) + self.bias
        activations = tf.concat([
            tf.keras.activations.tanh(potentials[:, 0:self.n_outputs_per_activation]),
            tf.keras.activations.elu(potentials[:, self.n_outputs_per_activation:])
        ], axis=1)

        return activations
