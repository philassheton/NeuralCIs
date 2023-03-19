from neuralcis._data_saver import _DataSaver
from neuralcis._fifty_fifty_layer import _FiftyFiftyLayer
import neuralcis.common as common

from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
import numpy as np

# typing imports
from typing import Optional, Tuple
from neuralcis.common import Samples, NetInputs, NetOutputs, NodesInLayer
from neuralcis.common import NetOutputBlob, NetTargetBlob
from neuralcis.common import TrainingBatches
from tensor_annotations.tensorflow import Tensor1, Tensor2
import tensor_annotations.tensorflow as ttf
tf32 = ttf.float32


class _SimulatorNet(ABC, _DataSaver):
    """Train a multilayer perceptron based on data generated by simulation.

    This is where the actual neural networks live, and at some point will
    also be added features for automatically searching for best
    hyperparameters, etc.  The following functions must be implemented:

    :func simulate_training_data: A `tf.function` that takes as
        input a single `int` and generates a tensor of input data and
        (optionally) a tensor of target outputs (if those are required by
        the loss function). If no target outputs are needed, they should be
        replaced with a None value.
    :func get_validation_set: A function that takes no arguments and returns
        a static validation set in the same format as from
        `self.simulate_training_data`.  This needs to be a function as the
        validation set itself might be transformed by a non-static transform
        (such as another neural network) before being used here.  It must
        also be a `tf.function`.
    :func loss: A function that takes the output from
        `self.run_net_during_training` and the target values from
        `self.sampling_distribution_fn` and calculates a loss value.

    The following function can be overridden if more than just the network
    outputs are needed to calculate the loss:

    :func run_net_during_training: OPTIONAL - A `tf.function` that takes as
        input a 2D samples x net inputs `Tensor` (plus an optional
        "training" bool -- default True) and applies the inputs to the
        network.  This only need be overridden if it is necessary to pull
        other numbers from the neural network than just its outputs, in
        order to calculate the loss.
    :func compute_optimum_loss: OPTIONAL - A `tf.function` that takes no
        arguments and computes an estimate for the optimum loss value for
        the validation set.

    The `_SimulatorNet` can be constructed without passing parameters, but
    the following parameters in the constructor allow features of the
    network to be tweaked (note that it will infer the number of inputs
    by pulling a test sample from `self.simulate_training_data`):

    :param num_outputs: An int, number outputs for the network (defaults
        to 1).
    :param num_hidden_layers: An int (optional), number of hidden layers in
        the network (default 3).
    :param num_neurons_per_hidden_layer: An int (optional), number of
        neurons per hidden layer (default 100).
    :param filename: A string (optional), if the network has been fit
        previously, and weights were saved, they can be loaded in the
        constructor by passing in the filename here.
    """
    def __init__(
            self,
            num_outputs: int = 1,
            num_hidden_layers: int = common.NUM_HIDDEN_LAYERS,
            num_neurons_per_hidden_layer: int = common.NEURONS_PER_LAYER,
            filename: str = ""
    ) -> None:

        self.net = self.create_net(num_outputs,
                                   num_hidden_layers,
                                   num_neurons_per_hidden_layer)

        # not filled in at init because it is slow / not always needed, so it
        #   needs to be explicitly filled by calling precompute_optimum_loss
        self.validation_optimum_loss = tf.Variable(np.nan)

        self.optimizer = tf.keras.optimizers.Nadam()

        # TODO: Wrap these lines of code up in a func so they are not repeated.
        validation_ins, validation_targets = self.get_validation_set()
        initial_validation_outs = self.run_net_during_training(
            self.net,
            validation_ins,
            tf.constant(False)
        )
        initial_loss = self.loss(
            initial_validation_outs,
            validation_targets
        )

        self.validation_loss_so_far = tf.Variable(initial_loss)
        self.batches_since_optimum = tf.Variable(0)

        _DataSaver.__init__(
            self,
            filename,
            instance_tf_variables_to_save=[
                "validation_loss_so_far"
            ],
            net_with_weights_to_save=self.net
        )

    ###########################################################################
    #
    #  Abstract methods
    #
    ###########################################################################

    @abstractmethod
    def simulate_training_data(
            self,
            n: ttf.int32
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], Optional[NetTargetBlob]]:

        """Generate net input samples and output targets.

        Must be a `tf.function`.

        :param n: An `int`, number of samples to generate.
        :return: A tuple containing two elements: (1) A 2D `Tensor` of
        samples x network inputs and (2) whatever target values are needed
        by the loss function (or None).
        """

    @abstractmethod
    def get_validation_set(
            self
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], Optional[NetTargetBlob]]:

        """Return the validation set.

        Must be a `tf.function`.

        Should return simulations in the same format as
        `sampling_distribution_fn`.

        :return: A tuple containing two elements: (1) A 2D `Tensor` of
        samples x network inputs and (2) whatever target values are needed
        by the loss function (or None).
        """

    @abstractmethod
    def loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: Optional[NetTargetBlob] = None
    ) -> ttf.float32:

        """Calculate the loss.

        Must be a `tf.function`.

        :param net_outputs: Outputs from the network in whatever format they
            are returned by `self.run_net` (generally a 2D samples x outputs
            `Tensor`, but can be changed by overriding the function).
        :param target_outputs: Whatever target data is output by
            `self.simulate_training_data`.
        :return: A `tf.float32` value.
        """

    ###########################################################################
    #
    #  Methods which might be overridden for better features.
    #
    ###########################################################################

    @tf.function
    def run_net_during_training(
            self,
            net: tf.keras.Model,
            net_inputs: Tensor2[tf32, Samples, NetInputs],
            training: ttf.bool = tf.constant(True)
    ) -> NetOutputBlob:                                                        # type: ignore

        """Generate from the network whatever outputs needed by `self.loss`.

        Can be overridden if the loss function needs more than just the
        outputs of the network to calculate the loss.

        :param net: The underlying Keras network to be run.
        :param net_inputs: A 2D samples x inputs `Tensor` of inputs.
        :param training: A bool that is passed into the Sequential object,
            defaults to True (since we only run this function in training
            mode).  Is not used except to pass to the Keras model.
        :return: Whatever values from the network that are needed by
            `self.loss` in order to be able to calculate the loss.
        """

        return net(net_inputs, training=training)

    @tf.function
    def compute_optimum_loss(self) -> ttf.float32:

        """Compute an estimate of the optimum loss for the validation set.

        If this is not possible, the default behaviour in this version
        (which returns `np.nan`) will suppress use of an optimum loss.

        :return: A `tf.float32` estimate of the optimum loss value.
        """

        return tf.constant(np.nan)

    ###########################################################################
    #
    #  Non-Tensorflow members
    #
    ###########################################################################

    def create_net(
            self,
            num_outputs: int,
            num_hidden_layers: int,
            num_neurons_per_hidden_layer: int
    ) -> tf.keras.Sequential:

        num_neurons_of_each_type = int(num_neurons_per_hidden_layer / 2)

        net = tf.keras.models.Sequential()
        for i in range(num_hidden_layers):
            net.add(_FiftyFiftyLayer(num_neurons_of_each_type))
        net.add(tf.keras.layers.Dense(
            num_outputs,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer="zeros"
        ))

        # Do this to force it to instantiate some weights
        net_ins, net_outs = self.simulate_training_data(tf.constant(2))
        net(net_ins)

        return net

    def set_learning_rate(
            self,
            learning_rate: Optional[float] = None
    ) -> None:

        if learning_rate is None:
            learning_rate = self.optimizer.learning_rate
        self.optimizer.lr.assign(learning_rate)

    def fit(
            self,
            max_epochs:
                int = common.MAX_EPOCHS,
            minibatch_size:
                int = common.MINIBATCH_SIZE,
            learning_rate_initial:
                float = common.LEARNING_RATE_INITIAL,
            divide_after_flattening_for:
                int = common.DIVIDE_AFTER_FLATTENING_FOR,
            target_validation_loss_sd:
                float = common.TARGET_VALIDATION_LOSS_SD,
            precompute_optimum_loss:
                bool = False
    ) -> None:

        if precompute_optimum_loss:
            self.precompute_optimum_loss()

        learning_rate = learning_rate_initial
        for epoch in range(max_epochs):
            self.set_learning_rate(learning_rate)
            learning_rate_check = self.optimizer.learning_rate.numpy()
            print(
                "Learning rate %f -- loss to beat is %f" %
                (learning_rate_check, self.validation_loss_so_far.numpy())     # type: ignore
            )
            validation_losses = self.fit_tf(
                num_minibatches_per_batch=tf.constant(
                    common.MINIBATCHES_PER_BATCH
                ),
                num_batches=tf.constant(common.BATCHES_PER_EPOCH),
                minibatch_size=tf.constant(minibatch_size)
            )

            validation_sd = tfp.stats.stddev(validation_losses)
            if validation_sd < target_validation_loss_sd:
                break

            if self.batches_since_optimum > divide_after_flattening_for:       # type: ignore
                learning_rate = learning_rate / 2.

    def precompute_optimum_loss(self) -> None:
        print("Calculating optimum loss")
        optimum_loss = self.compute_optimum_loss()
        print("Calculated optimum loss")

        self.set_validation_optimum_loss(optimum_loss)

    ###########################################################################
    #
    #  Tensorflow functions
    #
    ###########################################################################

    # a "batch" is the amount it does before reporting learning rates etc
    @tf.function
    def fit_tf(
            self,
            num_minibatches_per_batch: ttf.int32,
            num_batches: ttf.int32,
            minibatch_size: ttf.int32
    ) -> Tensor1[tf32, TrainingBatches]:

        optimum_text = tf.cond(
            tf.math.is_nan(self.validation_optimum_loss),
            lambda: "",
            lambda: tf.strings.format(
                " (vs optimum: {})",
                self.validation_optimum_loss
            )
        )
        validation_losses = tf.TensorArray(tf.float32, num_batches,
                                           element_shape=())
        for batch in tf.range(num_batches):
            for minibatch in tf.range(num_minibatches_per_batch):
                net_ins, net_targets = self.simulate_training_data(
                    minibatch_size
                )
                loss, grads = self.loss_and_gradient(
                    net_ins,
                    net_targets,
                    tf.constant(True)
                )
                self.optimizer.apply_gradients(zip(
                    grads,
                    self.net.trainable_weights
                ))

            validation_ins, validation_targets = self.get_validation_set()
            validation_outs = self.run_net_during_training(
                self.net,
                validation_ins,
                tf.constant(False)
            )
            validation_loss = self.loss(validation_outs, validation_targets)
            validation_losses = validation_losses.write(batch, validation_loss)

            tf.print(tf.strings.format(
                "Training loss: {}{}",
                (validation_loss, optimum_text)
            ))

            batches_since_optimum = tf.cond(
                validation_loss < self.validation_loss_so_far,                 # type: ignore
                lambda: 0,
                lambda: self.batches_since_optimum + 1                         # type: ignore
            )
            validation_loss_so_far = tf.math.minimum(
                self.validation_loss_so_far,
                validation_loss
            )

            self.batches_since_optimum.assign(batches_since_optimum)
            self.validation_loss_so_far.assign(validation_loss_so_far)

        return validation_losses.stack()

    @tf.function
    def call_tf(
            self,
            net_ins: Tensor2[tf32, Samples, NetInputs]
    ) -> Tensor2[tf32, Samples, NetOutputs]:

        return self.net(net_ins)

    @tf.function
    def loss_and_gradient(
            self,
            net_ins: Tensor2[tf32, Samples, NetInputs],
            net_targets: Optional[NetTargetBlob],
            training: ttf.bool = tf.constant(True)
    ) -> Tuple[ttf.float32, list]:

        with tf.GradientTape() as tape2:                                       # type: ignore
            tape2.watch(self.net.trainable_weights)
            net_outs = self.run_net_during_training(
                self.net,
                net_ins,
                training
            )
            loss = self.loss(net_outs, net_targets)

        gradient = tape2.gradient(loss, self.net.trainable_weights)
        return loss, gradient

    ###########################################################################
    #
    #  Extra functions for analysing the network.
    #
    ###########################################################################

    @tf.function
    def get_layer_outputs_variance(
            self,
            layer_out: Tensor2[tf32, Samples, NodesInLayer]
    ) -> Tensor1[tf32, NodesInLayer]:
        return tfp.stats.variance(layer_out, sample_axis=0)

    def get_neuron_variances_for_validation_set(self) -> list:
        extractor = tf.keras.models.Model(
            inputs=self.net.input,
            outputs=[layer.output for layer in self.net.layers]
        )
        validation_ins, validation_outs = self.get_validation_set()
        layer_outs = extractor(validation_ins)
        layer_variances = [
            tfp.stats.variance(out, sample_axis=0) for out in layer_outs
        ][0:-1]

        return layer_variances

    def set_validation_optimum_loss(
            self,
            validation_optimum_loss: ttf.float32
    ) -> None:

        self.validation_optimum_loss.assign(validation_optimum_loss)