from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp

from typing import List
from tensor_annotations.tensorflow import Tensor2
import tensor_annotations.tensorflow as ttf
from neuralcis.common import Samples, LayerInputs, LayerOutputs
tf32 = ttf.float32


class _SimNetLayer(tf.keras.layers.Layer, ABC):
    def __init__(self, num_outputs: int) -> None:
        super().__init__()
        self.num_outputs = num_outputs
        self.kernel = None
        self.bias = None

    def build(self, input_shape: List) -> None:
        num_inputs = int(input_shape[-1])
        self.kernel = self.add_weight(
            "kernel",
            shape=(num_inputs, self.num_outputs),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        self.bias = self.add_weight(
            "bias",
            shape=(self.num_outputs,),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.1,
                                                            maxval=0.1),
            trainable=True
        )

    @abstractmethod
    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tensor2[tf32, Samples, LayerInputs]:
        pass


class _LinearLayer(_SimNetLayer):
    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tensor2[tf32, Samples, LayerInputs]:
        return tf.linalg.matmul(inputs, self.kernel) + self.bias


class _FiftyFiftyLayer(_SimNetLayer):
    def __init__(self, num_outputs: int) -> None:
        _SimNetLayer.__init__(self, num_outputs)
        assert num_outputs % 2 == 0
        self.num_outputs_per_activation = num_outputs // 2

    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs]
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        potentials = tf.linalg.matmul(inputs, self.kernel) + self.bias
        tanh_outputs = potentials[:, 0:self.num_outputs_per_activation]
        elu_outputs = potentials[:, self.num_outputs_per_activation:]
        activations = tf.concat([
            tf.keras.activations.tanh(tanh_outputs),
            tf.keras.activations.elu(elu_outputs)
        ], axis=1)

        return activations


class _MultiplyerLayer(_SimNetLayer):
    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs]
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        potentials = tf.linalg.matmul(inputs, self.kernel) + self.bias

        outputs = potentials * inputs
        mean = tf.math.reduce_mean(outputs, axis=1)
        sd = tf.math.reduce_std(outputs, axis=1)
        outputs = (outputs - mean[:, None]) / sd[:, None]

        outputs = outputs + inputs

        return outputs


class _MonotonicLayer(_SimNetLayer):
    @tf.function
    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        kernel = tf.math.exp(self.kernel)
        bias = tf.math.exp(self.bias)
        potentials = tf.linalg.matmul(inputs, kernel) + bias
        activations = tf.keras.activations.sigmoid(potentials)

        return activations


class DoubleTriangularLayer(_SimNetLayer):
    def __init__(self, num_outputs: int) -> None:
        super().__init__(num_outputs)
        self.num_outputs = num_outputs
        self.lower = None
        self.upper = None
        self.bias = None

    def build(self, input_shape: list) -> None:
        num_inputs = int(input_shape[-1])
        num_weights = num_inputs * (num_inputs + 1) // 2
        assert num_inputs == self.num_outputs

        self.lower = self.add_weight(
            'lower',
            shape=(num_weights, ),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        self.upper = self.add_weight(
            'upper',
            shape=(num_weights, ),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        self.bias = self.add_weight(
            'bias',
            shape=(self.num_outputs, ),
            initializer=tf.keras.initializers.RandomUniform(),
            trainable=True
        )

    def call(self, inputs):
        lower = tfp.math.fill_triangular(self.lower)
        upper = tfp.math.fill_triangular(self.upper, upper=True)
        outputs = tf.linalg.matmul(inputs, lower)
        outputs = tf.linalg.matmul(outputs, upper)
        outputs = outputs + self.bias[None, :]
        outputs = tf.keras.activations.elu(outputs)

        return outputs


class _DefaultIn(_LinearLayer):
    pass


class _DefaultHid(_MultiplyerLayer):
    pass


class _DefaultOut(_LinearLayer):
    pass
