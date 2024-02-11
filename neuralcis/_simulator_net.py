from neuralcis._data_saver import _DataSaver
from neuralcis._fifty_fifty_layer import _FiftyFiftyLayer
import neuralcis.common as common

from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
import numpy as np

# typing imports
from typing import Optional, Tuple, List, Sequence, Union
from neuralcis.common import Samples, NetInputs, NetOutputs, NodesInLayer
from neuralcis.common import NetOutputBlob, NetTargetBlob
from tensor_annotations.tensorflow import Tensor1, Tensor2
import tensor_annotations.tensorflow as ttf
tf32 = ttf.float32


class _SimulatorNet(tf.keras.Model, ABC, _DataSaver):
    """Train a (or several) multilayer perceptrons based on data generated by
    simulation.

    This is where the actual neural networks live, and at some point will
    also be added features for automatically searching for best
    hyperparameters, etc.  The following functions must be implemented:

    :func simulate_training_data: A `tf.function` that takes as
        input a single `int` and generates for each net a tensor of input data
        (all wrapped up into a list of input data tensors) and
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

    :param num_outputs: A list of ints, number outputs for each network.
        Defaults to [1].
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
            num_outputs: Sequence[int] = (1,),
            num_hidden_layers: int = common.NUM_HIDDEN_LAYERS,
            num_neurons_per_hidden_layer: int = common.NEURONS_PER_LAYER,
            filename: str = "",
            subobjects_to_save: dict = None,
            *model_args,
            **model_kwargs,
    ) -> None:

        tf.keras.Model.__init__(self, *model_args, **model_kwargs)

        self.nets = self.create_nets(num_outputs,
                                     num_hidden_layers,
                                     num_neurons_per_hidden_layer)

        trainable_weights = [net.trainable_weights for net in self.nets]
        self.train_weights = [w for ws in trainable_weights for w in ws]

        # not filled in at init because it is slow / not always needed, so it
        #   needs to be explicitly filled by calling precompute_optimum_loss
        self.validation_optimum_loss = tf.Variable(np.nan)

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

        # TODO: Must figure out something more elegant than this.  Currently
        #       we are passing in a dummy dataset that generates a single
        #       random int as dataset and then we generate our own dataset
        #       inside of train_step and ignore the "random" dataset.
        #  -- see also tf.data.experimental.RandomDataset, which this function
        #       constructs (constructing directly is deprecated) and that
        #       give some clues about how to do this correctly.
        self.dummy_dataset = tf.data.Dataset.random()

        _DataSaver.__init__(
            self,
            filename,
            instance_tf_variables_to_save=[
                "validation_loss_so_far"
            ],
            nets_with_weights_to_save=self.nets,
            subobjects_to_save=subobjects_to_save
        )

        self.compile()

    ###########################################################################
    #
    #  Abstract methods
    #
    ###########################################################################

    @abstractmethod
    def simulate_training_data(
            self,
            n: ttf.int32
    ) -> Tuple[Sequence[Tensor2[tf32, Samples, NetInputs]],
               Optional[NetTargetBlob]]:

        """Generate net input samples and output targets.

        Must be a `tf.function`.

        :param n: An `int`, number of samples to generate.
        :return: A tuple containing two elements: (1) A list of 2D `Tensor`s
        of samples x network inputs (one for each net) and (2) whatever target
        values are needed by the loss function (or None).
        """

    @abstractmethod
    def get_validation_set(
            self
    ) -> Tuple[Sequence[Tensor2[tf32, Samples, NetInputs]],
               Optional[NetTargetBlob]]:

        """Return the validation set.

        Must be a `tf.function`.

        Should return simulations in the same format as
        `sampling_distribution_fn`.

        :return: A tuple containing two elements: (1) A list of 2D `Tensor`s
        of samples x network inputs (one for each net) and (2) whatever target
        values are needed by the loss function (or None).
        """

    @abstractmethod
    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: Optional[NetTargetBlob] = None
    ) -> ttf.float32:

        """Calculate the loss.

        Must be a `tf.function`.

        :param net_outputs: Outputs from the networks in whatever format they
            are returned by `self.run_net` (generally a tuple of 2D samples x
            outputs `Tensor`s, but can be changed by overriding the function).
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
    def run_nets_during_training(
            self,
            nets: Sequence[tf.keras.Model],
            net_inputs: Sequence[Tensor2[tf32, Samples, NetInputs]],
            training: ttf.bool = tf.constant(True)
    ) -> NetOutputBlob:                                                        # type: ignore

        """Generate from the network whatever outputs needed by `self.get_loss`.

        Can be overridden if the loss function needs more than just the
        outputs of the network to calculate the loss.

        :param nets: The underlying Keras networks to be run (a list).
        :param net_inputs: A list of 2D samples x inputs `Tensor`s of inputs
            (one for each net)...
        :param training: A bool that is passed into the Sequential object,
            defaults to True (since we only run this function in training
            mode).  Is not used except to pass to the Keras model.
        :return: Whatever values from the network that are needed by
            `self.get_loss` in order to be able to calculate the loss.
        """

        return self.call_tf(net_inputs, training=training)

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
    #  Methods overridden from tf.keras.Model
    #
    ###########################################################################

    def fit(
            self,
            steps_per_epoch: int = common.STEPS_PER_EPOCH,
            epochs: int = common.EPOCHS,
            verbose: Union[int, str] = 'auto',
            learning_rate_initial: int = common.LEARNING_RATE_INITIAL,
            learning_rate_half_life_epochs: int
                                       = common.LEARNING_RATE_HALF_LIFE_EPOCHS,
            callbacks: Sequence[tf.keras.callbacks.Callback] = None,
            *args
    ) -> None:

        # TODO: this is rather ugly, effectively making any call from the
        #       superclass to this function break.  Rethink!
        assert len(args) == 0   # Cannot use the usual fit args here!!

        def learning_rate(epoch):
            return learning_rate_initial * 2**(-epoch /
                                               learning_rate_half_life_epochs)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(learning_rate)

        if callbacks is None:
            callbacks = [lr_scheduler]
        else:
            callbacks = [lr_scheduler] + list(callbacks)

        super().fit(x=self.dummy_dataset,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=verbose,
                    callbacks=callbacks)

    def compile(self, optimizer='nadam', *args, **kwargs):
        super().compile(optimizer, *args, **kwargs)

    def train_step(self, _):
        net_ins, net_targets = self.simulate_training_data(
            common.MINIBATCH_SIZE
        )
        loss, grads = self.loss_and_gradient(
            net_ins,
            net_targets,
            tf.constant(True)
        )
        self.optimizer.apply_gradients(zip(
            grads,
            self.train_weights
        ))

        self.loss_tracker.update_state(loss)

        return {'loss': self.loss_tracker.result()}

    def test_step(self, data):
        return {'loss_val': self.validation_loss()}

    ###########################################################################
    #
    #  Non-Tensorflow members
    #
    ###########################################################################

    def create_nets(
            self,
            num_outputs: Sequence[int],
            num_hidden_layers: int,
            num_neurons_per_hidden_layer: int
    ) -> List[tf.keras.Sequential]:

        return [self.create_net(i, n_out,
                                num_hidden_layers,
                                num_neurons_per_hidden_layer)
                for i, n_out in enumerate(num_outputs)]

    def create_net(
            self,
            net_num: int,
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
        net(net_ins[net_num])

        return net

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

    @property
    def metrics(self):
        return [self.loss_tracker]

    @tf.function
    def call_tf(
            self,
            net_ins: Sequence[Tensor2[tf32, Samples, NetInputs]],
            training: bool = False
    ) -> Tensor2[tf32, Samples, NetOutputs]:

        outputs = [net(ins, training=training)
                   for net, ins in zip(self.nets, net_ins)]
        return tf.concat(outputs, axis=1)

    @tf.function
    def validation_loss(self):
        validation_ins, validation_targets = self.get_validation_set()
        validation_outs = self.run_nets_during_training(
            self.nets,
            validation_ins,
            tf.constant(False)
        )
        validation_loss = self.get_loss(validation_outs, validation_targets)

        return validation_loss

    @tf.function
    def loss_and_gradient(
            self,
            net_ins: Sequence[Tensor2[tf32, Samples, NetInputs]],
            net_targets: Optional[NetTargetBlob],
            training: ttf.bool = tf.constant(True)
    ) -> Tuple[ttf.float32, list]:

        with tf.GradientTape() as tape2:                                       # type: ignore
            tape2.watch(self.train_weights)
            net_outs = self.run_nets_during_training(
                self.nets,
                net_ins,
                training
            )
            loss = self.get_loss(net_outs, net_targets)

        gradient = tape2.gradient(loss, self.train_weights)
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

    def get_neuron_variances_for_validation_set(self, net_index) -> list:
        extractor = tf.keras.models.Model(
            inputs=self.nets[net_index].input,
            outputs=[layer.output for layer in self.nets[net_index].layers]
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
