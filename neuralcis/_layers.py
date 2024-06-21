from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from typing import List
from tensor_annotations.tensorflow import Tensor2

import neuralcis.common as common

# Typing
from typing import List, Tuple
from tensor_annotations.tensorflow import Tensor1, Tensor2
import tensor_annotations.tensorflow as ttf
from neuralcis.common import Samples, LayerInputs, LayerOutputs
tf32 = ttf.float32


@tf.function
def scaled_tanh(x):
    return tf.keras.activations.tanh(x) * common.TANH_MULTIPLIER


class _SimNetLayer(tf.keras.layers.Layer, ABC):
    must_have_same_inputs_as_outputs = False

    def __init__(self, num_outputs: int) -> None:
        super().__init__()
        self.num_outputs = num_outputs
        self.num_inputs = None
        self.kernel = None
        self.bias = None

    def build(self, input_shape: List) -> None:
        self.num_inputs = int(input_shape[-1])
        if self.must_have_same_inputs_as_outputs:
            self.check_inputs_and_outputs_match()

        kernel_min, kernel_max = self.kernel_init_min_max(self.num_inputs,
                                                          self.num_outputs)

        self.kernel = self.add_weight(
            "kernel",
            shape=(self.num_inputs, self.num_outputs),
            initializer=tf.keras.initializers.RandomUniform(minval=kernel_min,
                                                            maxval=kernel_max),
            trainable=True,
        )
        self.bias = self.add_weight(
            "bias",
            shape=(self.num_outputs,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def check_inputs_and_outputs_match(self):
        if self.num_inputs != self.num_outputs:
            class_name = self.__class__.__name__
            raise Exception(f"For a {class_name}, the input size must"
                            f" match the output size.  In this case we have"
                            f" {self.num_inputs} inputs and {self.num_outputs}"
                            f" outputs.")

    # logic here is glorot should give us nice behaviour for relu, but we might
    #    have an activation function that gets saturated at those same weights,
    #    and anyway we will scale_weights once constructed, so we start with
    #    Glorot as a basic guide and then divide by 10 to get us something
    #    "safe".
    def kernel_init_min_max(
            self,
            fan_in: int,
            fan_out: int
    ) -> Tuple[float, float]:

        glorot_size = np.sqrt(6. / (fan_in + fan_out))
        glorot_safe = glorot_size / 10.
        return -glorot_safe, glorot_safe

    def scale_weights(self):
        print(f"Rescaling {self.__class__.__name__} weights:")
        initializer = _LayerInitializer(self)
        initializer.compile()
        initializer.fit()

        test_inputs = tf.random.normal((1000, self.num_inputs))
        test_outputs = self(test_inputs)
        print(f"Means: {tf.math.reduce_mean(test_outputs, 0).numpy()}")
        print(f"SDs:   {tf.math.reduce_std(test_outputs, 0).numpy()}")
        print(f"Mins:  {tf.math.reduce_min(test_outputs, 0).numpy()}")
        print(f"Maxes: {tf.math.reduce_max(test_outputs, 0).numpy()}")

    @abstractmethod
    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tensor2[tf32, Samples, LayerInputs]:
        pass


class _LayerInitializer(tf.keras.models.Model):
    def __init__(self, layer: _SimNetLayer):
        super().__init__()
        self.layer = layer
        self.dummy_dataset = tf.data.Dataset.random()
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    def train_step(self, _):
        # TODO: Replace this with a proper loss function
        weights = self.trainable_weights
        batch = common.MINIBATCH_SIZE
        inputs = tf.random.normal((batch, self.layer.num_inputs))
        with tf.GradientTape() as tape:
            tape.watch(weights)
            outputs = self.layer(inputs)
            variances = tf.math.reduce_variance(outputs, 0)
            means = tf.math.reduce_mean(outputs, 0)
            loss = tf.reduce_mean(
                tf.math.abs(tf.math.log(variances)) +                          # This will NOT be consistent across batches...
                tf.math.square(means)
            )
        gradients = tape.gradient(loss, weights)
        self.optimizer.apply_gradients(zip(gradients, weights))
        self.loss_tracker.update_state(loss)

        return {'loss': self.loss_tracker.result()}

    def compile(self, optimizer='sgd', *args, **kwargs):
        super().compile(optimizer, *args, **kwargs)

    def fit(self, *args) -> None:
        # TODO: this is rather ugly, effectively making any call from the
        #       superclass to this function break.  Rethink!
        assert len(args) == 0  # Cannot use the usual fit args here!!

        learning_rate_initial = common.LAYER_LEARNING_RATE_INITIAL
        learning_rate_half_life_epochs = common.LAYER_LEARNING_HALF_LIFE_EPOCHS
        steps_per_epoch = common.LAYER_LEARNING_STEPS_PER_EPOCH
        epochs = common.LAYER_LEARNING_EPOCHS

        def learning_rate(epoch):
            half_lives = epoch / learning_rate_half_life_epochs
            return learning_rate_initial * 2 ** -half_lives

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(learning_rate)

        super().fit(x=self.dummy_dataset,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=True,
                    callbacks=[lr_scheduler])


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
            scaled_tanh(tanh_outputs),
            tf.keras.activations.elu(elu_outputs)
        ], axis=1)

        return activations


class _MultiplyerLayer(_SimNetLayer):
    must_have_same_inputs_as_outputs = True

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


class _MonotonicLinearLayer(_SimNetLayer):
    @tf.function
    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        kernel = tf.math.exp(self.kernel)
        bias = self.bias
        potentials = tf.linalg.matmul(inputs, kernel) + bias

        return potentials

    def kernel_init_min_max(
            self,
            fan_in: int,
            fan_out: int
    ) -> Tuple[float, float]:

        neg, pos = super().kernel_init_min_max(fan_in, fan_out)
        return np.log(pos * .5), np.log(pos)                                   # Must be positive!


class _MonotonicTanhLayer(_MonotonicLinearLayer):
    @tf.function
    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        potentials = super().call(inputs)
        activations = scaled_tanh(potentials)

        return activations


class _DoubleTriangularLayer(_SimNetLayer):
    must_have_same_inputs_as_outputs = True

    def __init__(self, num_outputs: int) -> None:
        super().__init__(num_outputs)
        self.num_outputs = num_outputs
        self.lower = None
        self.upper = None
        self.bias = None

    def build(self, input_shape: list) -> None:
        self.num_inputs = int(input_shape[-1])
        self.check_inputs_and_outputs_match()
        num_weights = self.num_inputs * (self.num_inputs + 1) // 2
        assert self.num_inputs == self.num_outputs

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


_DefaultIn = _LinearLayer
_DefaultHid = _MultiplyerLayer
_DefaultOut = _LinearLayer

