from neuralcis._data_saver import _DataSaver
from neuralcis._layers import _SimNetLayer
from neuralcis._layers import _DefaultIn, _DefaultHid, _DefaultOut
from neuralcis import common, _layers

import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
import numpy as np
import collections
import inspect

from abc import ABC, abstractmethod

# typing imports
from typing import Optional, Tuple, Sequence, List, Dict, Union, Type
from neuralcis.common import Samples, NetInputs, NetOutputs, NodesInLayer
from neuralcis.common import NetInputBlob, NetOutputBlob, NetTargetBlob
from tensor_annotations.tensorflow import Tensor1, Tensor2
import tensor_annotations.tensorflow as ttf
tf32 = ttf.float32

LayerTypeOrTypes = Union[
    Type[_SimNetLayer],
    Sequence[Type[_SimNetLayer]],
]


class _SimulatorNet(_DataSaver, tf.keras.Model, ABC):
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
        `self.call_tf(training=True)` and the target values from
        `self.sampling_distribution_fn` and calculates a loss value.

    The following function can be overridden if more than just the network
    outputs are needed to calculate the loss:

    :func call_tf_training: OPTIONAL - A `tf.function` that takes as
        input a 2D samples x net inputs `Tensor` and applies the inputs
        to the network.  If it is necessary to pull
        other numbers from the neural network than just its outputs, in
        order to calculate the loss, this can be overridden and these should
        be returned here..  Making this a separate function (as opposed to
        tf default of a call(training: bool) function) allows better type
        management in the cases we want the net to return different things
        during training.
    :func compute_optimum_loss: OPTIONAL - A `tf.function` that takes no
        arguments and computes an estimate for the optimum loss value for
        the validation set.

    The `_SimulatorNet` can be constructed without passing parameters, but
    the following parameters in the constructor allow features of the
    network to be tweaked (note that it will infer the number of inputs
    by pulling a test sample from `self.simulate_training_data`):

    :param num_outputs_for_each_net: A list of ints, number outputs for each
        network.  Defaults to [1].
    :param num_hidden_layers: An int (optional), number of hidden layers in
        the network (default 3).
    :param num_neurons_per_hidden_layer: An int (optional), number of
        neurons per hidden layer (default 100).
    :param first_layer_types: A type of a subclass of _SimNetLayer, or an
        array of such types.  If just a single type, this type will be
        applied for all underlying nets.
    :param hidden_layer_types: A type of a subclass of _SimNetLayer, or an
        array of such types.  If just a single type, this type will be
        applied for all underlying nets.
    :param output_layer_types: A type of a subclass of _SimNetLayer, or an
        array of such types.  If just a single type, this type will be
        applied for all underlying nets.
    :param filename: A string (optional), if the network has been fit
        previously, and weights were saved, they can be loaded in the
        constructor by passing in the filename here.
    """
    def __init__(
            self,
            num_outputs_for_each_net: Sequence[int] = (1,),
            num_hidden_layers: int = common.NUM_HIDDEN_LAYERS,
            num_neurons_per_hidden_layer: int = common.NEURONS_PER_LAYER,
            first_layer_type_or_types: LayerTypeOrTypes = _DefaultIn,
            hidden_layer_type_or_types: LayerTypeOrTypes = _DefaultHid,
            output_layer_type_or_types: LayerTypeOrTypes = _DefaultOut,
            layer_kwargs: Optional[Sequence[Dict]] = None,
            train_initial_weights: bool = False,
            filename: str = "",
            subobjects_to_save: dict = None,
            instance_tf_variables_to_save: Sequence[str] = (),
            *model_args,
            **model_kwargs,
    ) -> None:

        if layer_kwargs is None:
            layer_kwargs = [{} for o in num_outputs_for_each_net]

        tf.keras.Model.__init__(self, *model_args, **model_kwargs)

        self.nets, self.num_nets = self.create_nets(
            num_outputs_for_each_net,
            num_hidden_layers,
            num_neurons_per_hidden_layer,
            first_layer_type_or_types,
            hidden_layer_type_or_types,
            output_layer_type_or_types,
            layer_kwargs,
            train_initial_weights,
        )

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

        instance_tf_variables_to_save = list(instance_tf_variables_to_save)
        instance_tf_variables_to_save.append('validation_optimum_loss')
        _DataSaver.__init__(
            self,
            filename,
            instance_tf_variables_to_save=instance_tf_variables_to_save,
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
    ) -> Tuple[NetInputBlob, Optional[NetTargetBlob]]:

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
    ) -> Tuple[NetInputBlob, Optional[NetTargetBlob]]:

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
            are returned by `self.call_tf(training=True)` (generally a tuple
            of 2D samples x outputs `Tensor`s, but can be changed by overriding
            the function).
        :param target_outputs: Whatever target data is output by
            `self.simulate_training_data`.
        :return: A `tf.float32` value.
        """

    ###########################################################################
    #
    #  Methods which might be overridden for better features.
    #
    ###########################################################################

    def net_inputs(
            self,
            inputs: NetInputBlob,
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], ...]:

        return inputs

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

        use_decay = common.USE_DECAYING_LEARNING_RATE
        use_plateau = common.USE_DECREASE_LEARNING_RATE_ON_PLATEAU
        assert use_decay or use_plateau
        assert not (use_decay and use_plateau)

        print(f"Training {self.__class__.__name__}")

        if use_decay:
            def learning_rate(epoch):
                return learning_rate_initial * 2**(-epoch /
                                                learning_rate_half_life_epochs)
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
                learning_rate
            )

        elif use_plateau:
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss_val",
                factor=common.LEARNING_RATE_DECAY_RATIO_ON_PLATEAU,
                patience=common.LEARNING_RATE_PLATEAU_PATIENCE,
                min_lr=common.LEARNING_RATE_MINIMUM,
            )

        else:
            raise Exception("Should not even be possible to reach this!")

        if callbacks is None:
            callbacks = [lr_scheduler]
        else:
            callbacks = [lr_scheduler] + list(callbacks)

        return super().fit(x=self.dummy_dataset,
                           validation_data=self.dummy_dataset,
                           steps_per_epoch=steps_per_epoch,
                           validation_steps=1,
                           epochs=epochs,
                           verbose=verbose,
                           callbacks=callbacks)

    def compile(
            self,
            optimizer=None,
            *args,
            **kwargs
    ) -> None:

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(
                amsgrad=common.AMS_GRAD,
                learning_rate=common.LEARNING_RATE_INITIAL,
            )
        super().compile(optimizer, loss=None, *args, **kwargs)

    @tf.function
    def train_step(self, _):
        input_blob, targets = self.simulate_training_data(
            common.MINIBATCH_SIZE
        )
        loss, grads = self.loss_and_gradient(input_blob, targets)
        self.optimizer.apply_gradients(zip(grads, self.train_weights))

        self.loss_tracker.update_state(loss)

        return {'loss': self.loss_tracker.result()}

    def test_step(self, _):
        # TODO: At present will have to compile a different node for the
        #    validation runs, as tries to run the whole validation in one go,
        #    rather than breaking it into minibatches.  To break it into
        #    minibatches will need to write a custom dataloader that generates
        #    the data.
        return {'loss_val': self.validation_loss()}

    ###########################################################################
    #
    #  Non-Tensorflow members
    #
    ###########################################################################

    def create_nets(
            self,
            num_outputs_for_each_net: Sequence[int],
            num_hidden_layers: int,
            num_neurons_per_hidden_layer: int,
            first_layer_type_or_types: LayerTypeOrTypes,
            hidden_layer_type_or_types: LayerTypeOrTypes,
            output_layer_type_or_types: LayerTypeOrTypes,
            layer_kwargs: Sequence[Dict],
            train_initial_weights: bool,
    ) -> Tuple[
        List[tf.keras.Model],
        int,
    ]:

        self.num_nets = len(num_outputs_for_each_net)
        first_layer_types = self.repeat_layer_type_if_singleton(
            first_layer_type_or_types, self.num_nets,
        )
        hidden_layer_types = self.repeat_layer_type_if_singleton(
            hidden_layer_type_or_types, self.num_nets,
        )
        output_layer_types = self.repeat_layer_type_if_singleton(
            output_layer_type_or_types, self.num_nets,
        )

        self.nets = [self.create_net(n_out,
                                     num_hidden_layers,
                                     num_neurons_per_hidden_layer,
                                     first_layer_types[i],
                                     hidden_layer_types[i],
                                     output_layer_types[i],
                                     layer_kwargs[i])
                     for i, n_out in enumerate(num_outputs_for_each_net)]

        # Need to run some data through the nets to instantiate them.
        input_blob, targets = self.simulate_training_data(
            common.MINIBATCH_SIZE
        )
        self.call_tf(input_blob)
        if train_initial_weights:
            self.rescale_layer_weights()

        return self.nets, self.num_nets

    def rescale_layer_weights(self):
        my_class_name = self.__class__.__name__
        for i, net in enumerate(self.nets):
            print(f"\n\nRescaling weights in net {i} of {my_class_name}:")
            _layers.initialise_layers(net.layers)

    @staticmethod
    def create_net(
            num_outputs: int,
            num_hidden_layers: int,
            num_neurons_per_hidden_layer: int,
            input_layer_type: Type[_SimNetLayer],
            hidden_layer_type: Type[_SimNetLayer],
            output_layer_type: Type[_SimNetLayer],
            layer_kwargs: Dict,
    ) -> tf.keras.Sequential:

        net = tf.keras.models.Sequential()
        net.add(input_layer_type(num_neurons_per_hidden_layer, **layer_kwargs))
        for i in range(num_hidden_layers):
            net.add(hidden_layer_type(num_neurons_per_hidden_layer,
                                      **layer_kwargs))
        net.add(output_layer_type(num_outputs, **layer_kwargs))

        return net

    @staticmethod
    def repeat_layer_type_if_singleton(
            type_or_types: LayerTypeOrTypes,
            n: int,
    ) -> Sequence[Type[_SimNetLayer]]:

        if inspect.isclass(type_or_types):
            if issubclass(type_or_types, _SimNetLayer):
                return [type_or_types for _ in range(n)]
            else:
                raise Exception(
                    f'Layer type {type_or_types} is not a _SimNetLayer!!'
                )
        else:
            assert isinstance(type_or_types, collections.abc.Sequence)
            for t in type_or_types:
                assert inspect.isclass(t)
                assert issubclass(t, _SimNetLayer)
            assert len(type_or_types) == n
            return type_or_types

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
            input_blob: NetInputBlob,
    ) -> Tensor2[tf32, Samples, NetOutputs]:

        net_inputs = self.net_inputs(input_blob)
        return self._call_tf(net_inputs, training=False)

    @tf.function
    def call_tf_training(
            self,
            input_blob: NetInputBlob,
    ) -> NetOutputBlob:

        """Making this a separate function, rather than using the training
        argument, allows us to have different return types.  Because, in some
        networks, we want to override only what happens when we call the net
        during training (e.g. to return the Jacobians)."""

        net_inputs = self.net_inputs(input_blob)
        return self._call_tf(net_inputs, training=True)

    @tf.function
    def _call_tf(
            self,
            net_ins: Sequence[Tensor2[tf32, Samples, NetInputs]],
            training: bool,
    ) -> Tensor2[tf32, Samples, NetOutputs]:

        outputs = [net(ins, training=training)
                   for net, ins in zip(self.nets, net_ins)]
        return tf.concat(outputs, axis=1)

    @tf.function
    def validation_loss(self):
        validation_input_blob, validation_targets = self.get_validation_set()
        validation_outs = self.call_tf_training(validation_input_blob)
        validation_loss = self.get_loss(validation_outs, validation_targets)

        return validation_loss

    @tf.function
    def loss_and_gradient(
            self,
            input_blob: NetInputBlob,
            targets: Optional[NetTargetBlob]
    ) -> Tuple[ttf.float32, list]:

        with tf.GradientTape() as tape2:                                       # type: ignore
            tape2.watch(self.train_weights)
            net_outs = self.call_tf_training(input_blob)
            loss = self.get_loss(net_outs, targets)
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
        validation_input_blob, validation_outs = self.get_validation_set()
        validation_ins = self.net_inputs(validation_input_blob)[net_index]
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
