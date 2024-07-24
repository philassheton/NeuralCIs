import tensorflow as tf
from copy import deepcopy

from typing import List


class _ReduceLROnPlateauTrackBest(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(
            self,
            tensors_to_track: List[tf.Tensor],
            *args,
            **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.tensors_to_track = tensors_to_track
        self.tensor_copies = [deepcopy(t) for t in self.tensors_to_track]

    def on_epoch_end(
            self,
            epoch,
            logs=None,
    ):

        current = logs.get(self.monitor)
        best_before_call = self.best
        wait_before_call = self.wait

        super().on_epoch_end(epoch, logs)
        if self.in_cooldown():
            return

        if self.monitor_op(current, best_before_call):
            for tensor, copy in zip(self.tensors_to_track, self.tensor_copies):
                copy.assign(tensor)
        elif wait_before_call + 1 >= self.patience:
            tf.print(f"Restoring old ({best_before_call} beats {current})")
            for tensor, copy in zip(self.tensors_to_track, self.tensor_copies):
                tensor.assign(copy)
