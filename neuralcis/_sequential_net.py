from neuralcis._layers import layer_type_from_name

import tensorflow as tf
import pickle

# Typing
from typing import Sequence, Dict, Optional


WEIGHTS = "weights"
KWARGS = "kwargs"


class _SequentialNet:
    def __init__(
            self,
            filename: Optional[str] = None,
            **sequential_kwargs,
    ) -> None:

        self.sequential = None
        if filename is not None:
            self.sequential_kwargs = None
            self.load(filename)
        else:
            self.sequential_kwargs = sequential_kwargs
            self.construct_sequential(**self.sequential_kwargs)

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
        self.sequential(tf.ones((2, num_inputs)))

    def __call__(self, *args, **kwargs):
        return self.sequential.__call__(*args, **kwargs)

    def layers(self):
        return self.sequential.layers

    def trainable_weights(self):
        return self.sequential.trainable_weights

    def save(self, filename: str) -> None:
        with open(self.kwargs_filename(filename), "wb") as f:
            pickle.dump(self.sequential_kwargs, f)
        self.sequential.save_weights(self.weights_filename(filename))

    def load(self, filename: str) -> None:
        with open(self.kwargs_filename(filename), "rb") as f:
            self.sequential_kwargs = pickle.load(f)
        self.construct_sequential(**self.sequential_kwargs)
        self.sequential.load_weights(self.weights_filename(filename))

    @staticmethod
    def weights_filename(filename: str) -> str:
        return f"{filename} {WEIGHTS}"

    @staticmethod
    def kwargs_filename(filename: str) -> str:
        return f"{filename} {KWARGS}"
