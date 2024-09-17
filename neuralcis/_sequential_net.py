from neuralcis._layers import layer_type_from_name
from neuralcis import common

import tensorflow as tf
import pickle

# Typing
from typing import Sequence, Dict, Optional


WEIGHTS = "weights"
KWARGS = "kwargs"


class _SequentialNet:
    def __init__(
            self,
            input_batch_size,
            **sequential_kwargs,
    ) -> None:

        self.sequential = None
        self.sequential_kwargs = sequential_kwargs
        self.input_batch_size = input_batch_size

    def construct_sequential(
            self,
            num_inputs: int,
            num_outputs_per_layer: Sequence[int],
            layer_type_name_per_layer: Sequence[str],
            layer_kwargs_per_layer: Sequence[Dict],
    ) -> None:

        # TODO: chosen for now not to inherit from Sequential, so that the
        #       reference here to the sequential object can be easily replaced
        #       once load is called.  It might be worth thinking about
        #       rethinking the whole loading mechanism of the _DataSaver,
        #       so that weights are loaded at construction time, and then
        #       this would be not be necessary.
        self.sequential = tf.keras.Sequential()

        for layer_type_name, layer_outs, layer_kwargs in zip(
                layer_type_name_per_layer,
                num_outputs_per_layer,
                layer_kwargs_per_layer,
        ):

            layer_type = layer_type_from_name(layer_type_name)
            self.sequential.add(layer_type(layer_outs, **layer_kwargs))

        # Need to run some data through net to instantiate weights.
        self.sequential(tf.ones((self.input_batch_size, num_inputs)))

    def build(self) -> None:
        self.construct_sequential(**self.sequential_kwargs)

    def ensure_built(self) -> None:
        if self.sequential is None:
            self.build()

    @tf.function
    def __call__(self, *args, **kwargs):
        return self.sequential(*args, **kwargs)

    def layers(self):
        return self.sequential.layers

    # No need to be a tf.function as they are anyway stored in _SimulatorNet
    def trainable_weights(self):
        return self.sequential.trainable_weights

    def save(self, filename: str) -> None:
        with open(self.kwargs_filename(filename), "wb") as f:
            pickle.dump(self.sequential_kwargs, f)
        self.sequential.save_weights(self.weights_filename(filename))

    def load(self, filename: str) -> None:
        with open(self.kwargs_filename(filename), "rb") as f:
            self.sequential_kwargs = pickle.load(f)
        self.build()
        self.sequential.load_weights(self.weights_filename(filename))

    @staticmethod
    def weights_filename(filename: str) -> str:
        return f"{filename} {WEIGHTS}"

    @staticmethod
    def kwargs_filename(filename: str) -> str:
        return f"{filename} {KWARGS}"
