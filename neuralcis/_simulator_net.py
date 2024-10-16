from neuralcis._data_saver import _DataSaver
from neuralcis._sequential_net import _SequentialNet
from neuralcis import common, _layers, _callbacks

import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
import collections
import datetime

from abc import ABC, abstractmethod

# typing imports
from typing import Optional, Tuple, Sequence, List, Dict, Union
from neuralcis.common import Samples, NetInputs, NetOutputs, NodesInLayer
from neuralcis.common import NetInputBlob, NetOutputBlob, NetTargetBlob
from tensor_annotations.tensorflow import Tensor1, Tensor2
import tensor_annotations.tensorflow as ttf
tf32 = ttf.float32


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
    """

    # In a _SimulatorNet, each datapoint is fresh, so it makes more sense to
    # monitor our training based on the raw loss.
    loss_to_watch = "loss"
    absolute_loss_increase_tol = None
    relative_loss_increase_tol = None

    def __init__(
            self,
            num_inputs_for_each_net: Sequence[int],
            num_outputs_for_each_net: Sequence[int] = (1,),
            num_hidden_layers: int = common.NUM_HIDDEN_LAYERS,
            num_neurons_per_hidden_layer: int = common.NEURONS_PER_LAYER,
            input_layer_type_or_types: Union[str, Sequence[str]] =
                                                    common.LAYER_DEFAULT_IN,
            hidden_layer_type_or_types: Union[str, Sequence[str]] =
                                                    common.LAYER_DEFAULT_HID,
            output_layer_type_or_types: Union[str, Sequence[str]] =
                                                    common.LAYER_DEFAULT_OUT,
            layer_kwargs: Optional[Sequence[Dict]] = None,
            train_initial_weights: bool = False,
            batch_size: int = common.BATCH_SIZE,
            subobjects_to_save: dict = None,
            instance_tf_variables_to_save: Sequence[str] = (),
            *model_args,
            **model_kwargs,
    ) -> None:

        if layer_kwargs is None:
            layer_kwargs = [{} for _ in num_outputs_for_each_net]

        tf.keras.Model.__init__(self, *model_args, **model_kwargs)

        self.batch_size = batch_size
        self.nets, self.num_nets = self.create_nets(
            num_inputs_for_each_net,
            num_outputs_for_each_net,
            num_hidden_layers,
            num_neurons_per_hidden_layer,
            input_layer_type_or_types,
            hidden_layer_type_or_types,
            output_layer_type_or_types,
            batch_size,
            layer_kwargs,
        )

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.simnet_weights = None
        self.train_initial_weights = train_initial_weights

        _DataSaver.__init__(
            self,
            instance_tf_variables_to_save=instance_tf_variables_to_save,
            nets_with_weights_to_save=self.nets,
            subobjects_to_save=subobjects_to_save,
        )

    ###########################################################################
    #
    #  Abstract methods
    #
    ###########################################################################

    @abstractmethod
    def simulate_training_data(
            self,
    ) -> Tuple[NetInputBlob, NetTargetBlob]:

        """Generate net input samples and output targets.

        Must be a `tf.function`.

        :param n: An `int`, number of samples to generate.
        :return: A tuple containing two elements: (1) A list of 2D `Tensor`s
        of samples x network inputs (one for each net) and (2) whatever target
        values are needed by the loss function (or None).
        """

    @abstractmethod
    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: NetTargetBlob,
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

    @tf.function
    def net_inputs(
            self,
            inputs: NetInputBlob,
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], ...]:

        return inputs

    def get_ready_for_training(self) -> None:

        """Called at the start of the fit function.

        In _SimulatorNet, this builds all the net, lists its weights and
        compiles the model.  (Better to do this before training in case we
        make changes to the settings that impact the structure between
        instantiation and training -- e.g. this happens if the _DataSaver
        steps in to load the network structure from a file.

        :return: None
        """

        for net in self.nets:
            net.ensure_built()
        self.simnet_weights = self.get_simnet_weights()
        self.compile()

        # TODO: Either get rid of this altogether (not really needed any more
        #       since we don't have fancy layers any more) or at least move
        #       the whole initialisation thing into the _SequentialNet (also
        #       this bit still needs to be made parallelisable for GPU).
        if self.train_initial_weights:
            self.rescale_layer_weights()

    def erase_weights(self):
        for net in self.nets:
            net.erase_weights()

    ###########################################################################
    #
    #  Methods overridden from tf.keras.Model
    #
    ###########################################################################

    def fit(
            self,
            steps_per_epoch: int = common.STEPS_PER_EPOCH,
            epochs: int = common.EPOCHS,
            verbose: Union[int, str] = 2,
            learning_rate_initial: int = common.LEARNING_RATE_INITIAL,
            callbacks: Sequence[tf.keras.callbacks.Callback] = None,
            *args,
    ):

        # TODO: this is rather ugly, effectively making any call from the
        #       superclass to this function break.  Rethink!
        assert len(args) == 0   # Cannot use the usual fit args here!!

        self.get_ready_for_training()

        print(f"{datetime.datetime.now()}: Training {self.__class__.__name__}")

        lr_scheduler = _callbacks._ReduceLROnPlateauTrackBest(
            self.simnet_weights,
            monitor=self.loss_to_watch,
            learning_rate_initial=learning_rate_initial,
            factor=common.LEARNING_RATE_DECAY_RATIO_ON_PLATEAU,
            patience=common.LEARNING_RATE_PLATEAU_PATIENCE,
            min_lr=common.LEARNING_RATE_MINIMUM,
            absolute_loss_increase_tol=self.absolute_loss_increase_tol,
            relative_loss_increase_tol=self.relative_loss_increase_tol,
        )

        if callbacks is None:
            callbacks = [lr_scheduler]
        else:
            callbacks = [lr_scheduler] + list(callbacks)

        history = super().fit(x=self.dataset_generator(),
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              verbose=verbose,
                              callbacks=callbacks)

        print(f"{datetime.datetime.now()}: Restoring best")
        lr_scheduler.restore_best()

        return history

    def compile(
            self,
            optimizer=None,
            default_use_ams_grad=common.AMS_GRAD,
            *args,
            **kwargs,
    ) -> None:

        if optimizer is None:
            if default_use_ams_grad:
                optimizer = tf.keras.optimizers.Adam(amsgrad=True)
            else:
                optimizer = tf.keras.optimizers.Nadam()

        tf.keras.Model.compile(self, optimizer, loss=None, *args, **kwargs)

    @tf.function
    def train_step(self, data):
        input_blob, targets = data
        loss, grads = self.loss_and_gradient(input_blob, targets)
        self.optimizer.apply_gradients(zip(grads, self.simnet_weights))
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    def get_data_signature(self, data):
        if isinstance(data, tf.Tensor):
            return tf.TensorSpec.from_tensor(data)
        elif isinstance(data, collections.abc.Sequence):
            return tuple(self.get_data_signature(d) for d in data)
        else:
            raise ValueError("Data must be made of Tensors, Sequences or None")

    def data_generator(self) -> Tuple[NetInputBlob, NetTargetBlob]:
        while(True):
            yield self.simulate_training_data()

    def dataset_generator(self) -> tf.data.Dataset:
        sample_data = self.simulate_training_data()
        data_shape = self.get_data_signature(sample_data)
        dataset = tf.data.Dataset.from_generator(self.data_generator,
                                                 output_signature=data_shape)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    ###########################################################################
    #
    #  Non-Tensorflow members
    #
    ###########################################################################

    def create_nets(
            self,
            num_inputs_for_each_net: Sequence[int],
            num_outputs_for_each_net: Sequence[int],
            num_hidden_layers: int,
            num_neurons_per_hidden_layer: int,
            input_layer_type_or_types: Union[str, Sequence[str]],
            hidden_layer_type_or_types: Union[str, Sequence[str]],
            output_layer_type_or_types: Union[str, Sequence[str]],
            input_batch_size: int,
            layer_kwargs: Sequence[Dict],
    ) -> Tuple[
        List[_SequentialNet],
        int,
    ]:

        self.num_nets = len(num_outputs_for_each_net)
        input_layer_type_names = self.repeat_layer_type_if_singleton(
            input_layer_type_or_types, self.num_nets,
        )
        hidden_layer_type_names = self.repeat_layer_type_if_singleton(
            hidden_layer_type_or_types, self.num_nets,
        )
        output_layer_type_names = self.repeat_layer_type_if_singleton(
            output_layer_type_or_types, self.num_nets,
        )

        self.nets = [
            self.create_net(
                num_inputs=n_in,
                num_outputs=num_outputs_for_each_net[i],
                num_hidden_layers=num_hidden_layers,
                num_neurons_per_hidden_layer=num_neurons_per_hidden_layer,
                input_layer_type_name=input_layer_type_names[i],
                hidden_layer_type_name=hidden_layer_type_names[i],
                output_layer_type_name=output_layer_type_names[i],
                input_batch_size=input_batch_size,
                layer_kwargs=layer_kwargs[i]
            )
            for i, n_in in enumerate(num_inputs_for_each_net)
        ]

        return self.nets, self.num_nets

    def rescale_layer_weights(self):
        my_class_name = self.__class__.__name__
        for i, net in enumerate(self.nets):
            print(f"\n\nRescaling weights in net {i} of {my_class_name}:")
            _layers.initialise_layers(net.layers())

    @staticmethod
    def create_net(
            num_outputs: int,
            num_inputs: int,
            num_hidden_layers: int,
            num_neurons_per_hidden_layer: int,
            input_layer_type_name: str,
            hidden_layer_type_name: str,
            output_layer_type_name: str,
            input_batch_size: int,
            layer_kwargs: Dict,
    ) -> _SequentialNet:

        num_outputs_per_layer = [num_neurons_per_hidden_layer]
        layer_type_name_per_layer = [input_layer_type_name]
        layer_kwargs_per_layer = [layer_kwargs]

        for i in range(num_hidden_layers):
            num_outputs_per_layer.append(num_neurons_per_hidden_layer)
            layer_type_name_per_layer.append(hidden_layer_type_name)
            layer_kwargs_per_layer.append(layer_kwargs)

        num_outputs_per_layer.append(num_outputs)
        layer_type_name_per_layer.append(output_layer_type_name)
        layer_kwargs_per_layer.append(layer_kwargs)

        return _SequentialNet(
            input_batch_size,
            num_inputs=num_inputs,
            num_outputs_per_layer=num_outputs_per_layer,
            layer_type_name_per_layer=layer_type_name_per_layer,
            layer_kwargs_per_layer=layer_kwargs_per_layer,
        )

    @staticmethod
    def repeat_layer_type_if_singleton(
            type_or_types: Union[str, Sequence[str]],
            n: int,
    ) -> Sequence[str]:

        if isinstance(type_or_types, str):
            return [type_or_types for _ in range(n)]
        else:
            assert isinstance(type_or_types, collections.abc.Sequence)
            for t in type_or_types:
                assert isinstance(t, str)
            assert len(type_or_types) == n
            return type_or_types

    def get_simnet_weights(self) -> List[tf.Tensor]:
        simnet_weights = [net.trainable_weights() for net in self.nets]
        return [w for ws in simnet_weights for w in ws]

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
    def loss_and_gradient(
            self,
            input_blob: NetInputBlob,
            targets: NetTargetBlob,
    ) -> Tuple[ttf.float32, list]:

        with tf.GradientTape() as tape2:
            tape2.watch(self.simnet_weights)
            net_outs = self.call_tf_training(input_blob)
            loss = self.get_loss(net_outs, targets)
        gradient = tape2.gradient(loss, self.simnet_weights)

        return loss, gradient
