import neuralcis.common as common

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

# Typing
from typing import List, Tuple
from neuralcis.common import Samples, LayerInputs, LayerOutputs, Ys, Params
from tensor_annotations.tensorflow import Tensor2
import tensor_annotations.tensorflow as ttf

tf32 = ttf.float32


@tf.function
def scaled_tanh(x):
    return tf.keras.activations.tanh(x) * common.TANH_MULTIPLIER


class _SimNetLayer(tf.keras.layers.Layer, ABC):
    must_have_same_inputs_as_outputs = False
    initialization_step_size_multiplier = 1.

    def __init__(self, num_outputs: int, **layer_kwargs) -> None:
        super().__init__()
        self.num_outputs = num_outputs
        self.num_inputs = None
        self.kernel_raw = None                                                 # Critically, kernel_raw will be ignored by the initializer!!
        self.output_scaler = None                                              # The kernel scaler is just there to allow us to scale the individual channels in the initialization without adjusting individuals cells
        self.bias = None

    def weights_dims(self) -> Tuple[int, int]:
        return self.num_inputs, self.num_outputs

    def num_matmul_outputs(self) -> int:
        return self.num_outputs

    def initialisation_adjustables(self) -> Tuple[tf.Tensor, ...]:
        return self.output_scaler, self.bias

    def weights(self) -> Tuple[tf.Tensor, ...]:
        return self.kernel_raw * self.output_scaler, self.bias

    def build(self, input_shape: List) -> None:
        self.num_inputs = int(input_shape[-1])
        if self.must_have_same_inputs_as_outputs:
            self.check_inputs_and_outputs_match()

        W_num_in, W_num_out = self.weights_dims()
        kernel_min, kernel_max, bias_init = self.inits(W_num_in, W_num_out)

        self.kernel_raw = self.add_weight(
            "kernel_raw",
            shape=(W_num_in, W_num_out),
            initializer=tf.keras.initializers.RandomUniform(minval=kernel_min,
                                                            maxval=kernel_max),
            trainable=True,
        )
        self.output_scaler = self.add_weight(                                  # The output scaler exists to help with initialization
            "kernel_scaler",
            shape=(1, self.num_matmul_outputs()),
            initializer=tf.keras.initializers.Ones(),
            trainable=False,
        )
        self.bias = self.add_weight(
            "bias",
            shape=(W_num_out,),
            initializer=tf.keras.initializers.Constant(bias_init),
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
    def inits(
            self,
            fan_in: int,
            fan_out: int,
    ) -> Tuple[float, float, float]:

        glorot_size = np.sqrt(6. / (fan_in + fan_out))
        glorot_safe = glorot_size / 10.
        return -glorot_safe, glorot_safe, 0.

    @abstractmethod
    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tensor2[tf32, Samples, LayerInputs]:

        raise NotImplementedError()


class _LayerInitializer(tf.keras.models.Model):
    def __init__(
            self,
            layer: _SimNetLayer,
            previous_layers: List[_SimNetLayer],
    ) -> None:
        super().__init__()
        self.layer = layer
        self.previous_layers = previous_layers
        self.dummy_dataset = tf.data.Dataset.random()
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.layer_inputs = self.layer.num_inputs
        self.net_inputs = self.layer_inputs
        if len(self.previous_layers):
            self.net_inputs = self.previous_layers[0].num_inputs
        self.loss_rescale = (
            tf.math.sqrt(tf.cast(self.layer_inputs, tf.float32)) *             # Scaling up by number of inputs (sqrted) allows us to take big steps that are still not too big if there are very few inputs
            self.layer.initialization_step_size_multiplier
        )

    def train_step(self, _):
        weights = self.layer.initialisation_adjustables()
        batch = common.LAYER_LEARNING_BATCH_SIZE

        inputs = tf.random.uniform((batch, self.net_inputs), -1., 1.)
        for previous_layer in self.previous_layers:
            inputs = previous_layer(inputs)

        with (tf.GradientTape() as tape):
            tape.watch(weights)
            outputs = self.layer(inputs)
            variances = tf.math.reduce_variance(outputs, 0)
            means = tf.math.reduce_mean(outputs, 0)
            # TODO: First draft; replace this with a proper loss function
            loss = tf.reduce_mean(
                tf.math.abs(tf.math.log(variances * 3.)) +
                tf.math.square(means)
            )
            loss_scaled = (
                    loss *
                    self.loss_rescale *
                    self.layer.initialization_step_size_multiplier
            )

        gradients = tape.gradient(loss_scaled, weights)
        self.optimizer.apply_gradients(zip(gradients, weights))
        self.loss_tracker.update_state(loss)

        return {'loss': self.loss_tracker.result()}

    def compile(self, optimizer='sgd', *args, **kwargs):
        super().compile(optimizer, loss=None, *args, **kwargs)

    def fit(self, *args) -> None:
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


def initialise_layers(layers: List[_SimNetLayer]) -> None:
    num_inputs = layers[0].num_inputs
    test_values = tf.random.uniform((1000, num_inputs)) * 2. - 1
    for i, layer in enumerate(layers):
        previous_layers = layers[0:i]

        print(f"    Rescaling {layer.__class__.__name__} weights:")
        initializer = _LayerInitializer(layer, previous_layers)
        initializer.compile()
        initializer.fit()

        test_values = layer(test_values)
        print(f"        Mins:  {tf.math.reduce_min(test_values, 0).numpy()}")
        print(f"        Maxes: {tf.math.reduce_max(test_values, 0).numpy()}")
        print(f"        Means: {tf.math.reduce_mean(test_values, 0).numpy()}")
        print(f"        SDs:   {tf.math.reduce_std(test_values, 0).numpy()}")
        print(f"        ({layer.__class__.__name__} at layer {i})")


class _LinearLayer(_SimNetLayer):
    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tensor2[tf32, Samples, LayerInputs]:

        W, b = self.weights()
        return tf.linalg.matmul(inputs, W) + b


class _FiftyFiftyLayer(_SimNetLayer):
    def __init__(self, num_outputs: int) -> None:
        _SimNetLayer.__init__(self, num_outputs)
        assert num_outputs % 2 == 0
        self.num_outputs_per_activation = num_outputs // 2

    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        W, b = self.weights()
        potentials = tf.linalg.matmul(inputs, W) + b
        tanh_outputs = potentials[:, 0:self.num_outputs_per_activation]
        elu_outputs = potentials[:, self.num_outputs_per_activation:]
        activations = tf.concat([
            scaled_tanh(tanh_outputs),
            tf.keras.activations.elu(elu_outputs),
        ], axis=1)

        return activations


class _MultiplyerLayer(_SimNetLayer):
    must_have_same_inputs_as_outputs = True

    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        W, b = self.weights()
        potentials = tf.linalg.matmul(inputs, W) + b

        outputs = potentials * inputs
        mean = tf.math.reduce_mean(outputs, axis=1)
        sd = tf.math.reduce_std(outputs, axis=1)
        outputs = (outputs - mean[:, None]) / sd[:, None]

        outputs = outputs + inputs

        return outputs


class _MultiplyerWithSomeRelusLayer(_MultiplyerLayer):

    """_MultiplyerWithSomeRelusLayer multiplyer layer with a few ReLUs.

    This layer type exists because applying ReLU to *all* outputs of a layer
    leads very readily to a singular Jacobian matrix (for any data point that
    hits the left-hand-side of all of the ReLUs).  So, instead, we apply here
    the ReLU function to only the first `num_relu` outputs.  This allows us
    to produce relatively steep jumps in the output using the ReLU, but also
    get the benefits of the multiplyer layers."""

    def __init__(
            self,
            *args,
            num_relu: int = common.NUM_RELU_IN_MULTIPLYER_LAYER,
            **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.num_relu = num_relu

    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        num_relu = self.num_relu
        potentials = super().call(inputs)
        relu_activations = tf.keras.activations.relu(potentials[:, 0:num_relu])
        linear_activations = potentials[:, num_relu:]

        activations = tf.concat([relu_activations, linear_activations], axis=1)

        return activations


class _MonotonicLinearLayer(_SimNetLayer):
    initialization_step_size_multiplier = .1

    @tf.function
    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        Wlog, b = self.weights()
        W = tf.math.exp(Wlog)
        potentials = tf.linalg.matmul(inputs, W) + b
        activations = self.activation_function(potentials)

        return activations

    def activation_function(
            self,
            potentials: Tensor2[tf32, Samples, LayerOutputs],
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        return potentials

    def inits(
            self,
            fan_in: int,
            fan_out: int,
    ) -> Tuple[float, float, float]:

        neg, pos, b = super().inits(fan_in, fan_out)
        return np.log(pos * .5), np.log(pos), b                                # Must be positive!


class _MonotonicTanhLayer(_MonotonicLinearLayer):
    initialization_step_size_multiplier = 1.

    def activation_function(
            self,
            potentials: Tensor2[tf32, Samples, LayerOutputs],
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        return scaled_tanh(potentials)


class _MonotonicLeakyReluLayer(_MonotonicLinearLayer):

    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.leaky_relu = tf.keras.layers.LeakyReLU(common.LEAKY_RELU_SLOPE)

    def activation_function(
            self,
            potentials: Tensor2[tf32, Samples, LayerOutputs],
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        return self.leaky_relu(potentials)


class _MonotonicWithParamsTanhLayer(_MonotonicTanhLayer):

    def __init__(
            self,
            num_outputs: int,
            num_params: int,
    ) -> None:

        super().__init__(num_outputs + num_params)
        self.num_params = num_params
        self.num_mono_in = None
        self.num_mono_out = None
        self.num_virtual_weights = None
        self.output_shifter = None
        self.output_scaler = None

    def weights_dims(self):
        return self.num_params, (self.num_mono_in + 1) * self.num_mono_out     # +1 because we need an extra num_mono_out to generate virtual biases also

    def num_matmul_outputs(
            self,
    ) -> int:

        return self.num_mono_out

    def initialisation_adjustables(self) -> Tuple[tf.Tensor, ...]:
        return self.output_scaler, self.output_shifter

    def build(
            self,
            input_shape: List,
    ) -> None:

        self.num_mono_in = input_shape[-1] - self.num_params
        self.num_mono_out = self.num_outputs - self.num_params                 # For backward compatibility, self.num_outputs and self.num_inputs refer to the TOTAL number of inputs and outputs (including params)
        self.num_virtual_weights = self.num_mono_in * self.num_mono_out

        self.num_inputs = int(input_shape[-1])
        if self.must_have_same_inputs_as_outputs:
            self.check_inputs_and_outputs_match()

        kernel_rows, kernel_cols = self.weights_dims()
        virtual_kernel_min, virtual_kernel_max, bias_init = self.inits(
            self.num_mono_in,
            self.num_mono_out,
        )

        self.kernel_raw = self.add_weight(
            "kernel_raw",
            shape=(kernel_rows, kernel_cols),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.output_scaler = self.add_weight(
            # The output scaler exists to help with initialization
            "output_scaler",
            shape=(1, self.num_matmul_outputs()),
            initializer=tf.keras.initializers.Ones(),
            trainable=False,
        )
        self.output_shifter = self.add_weight(
            "output_shifter",
            shape=(1, self.num_matmul_outputs()),
            initializer=tf.keras.initializers.Zeros(),
            trainable=False,
        )
        self.bias = self.add_weight(
            "bias",
            shape=(kernel_cols,),
            # We use the kernel_min and kernel_max for the BIAS since, when
            #     combined with zero kernel as initialization, it will
            #     construct the virtual kernel within this min and max...
            initializer=tf.keras.initializers.RandomUniform(
                virtual_kernel_min,
                virtual_kernel_max,
            ),
            trainable=True,
        )

        # ... however the later elements of the BIAS *are* for the bias, not
        #     the kernel...
        self.bias[self.num_virtual_weights:].assign(
            self.bias[self.num_virtual_weights:] * 0. + bias_init
        )

    @tf.function
    def weights(
            self,
            params=None,
    ):

        if params is None:
            raise Exception("self.weights() called on with-params layer "
                            "without passing also params!")

        batch_size, _ = params.shape
        W, b = self.kernel_raw, self.bias
        logkernel_bias_flat = tf.linalg.matmul(params, W) + b
        logkernel = tf.reshape(
            logkernel_bias_flat[:, 0:self.num_virtual_weights],
            (batch_size, self.num_mono_in, self.num_mono_out),
        )
        bias = logkernel_bias_flat[:, self.num_virtual_weights:]

        return (
            logkernel + tf.math.log(self.output_scaler),
            bias + self.output_shifter
        )

    @tf.function
    def ins_mono_and_params(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tuple[Tensor2[tf32, Samples, Ys],
               Tensor2[tf32, Samples, Params]]:

        return inputs[:, 0:-self.num_params], inputs[:, -self.num_params:]     # type: ignore

    @tf.function
    def call(
            self,
            inputs: Tensor2[tf32, Samples, LayerInputs],
    ) -> Tensor2[tf32, Samples, LayerOutputs]:

        ins_mono, params = self.ins_mono_and_params(inputs)
        Wlog, b = self.weights(params)
        W = tf.math.exp(Wlog)

        # IMPORTANT: W is a 3d tensor of one weights matrix per data point
        potentials = tf.linalg.matmul(ins_mono[:, None, :], W)[:, 0, :] + b
        activations = self.activation_function(potentials)

        return tf.concat([activations, params], axis=1)


_DefaultIn = _LinearLayer
_DefaultHid = _MultiplyerLayer
_DefaultOut = _LinearLayer
