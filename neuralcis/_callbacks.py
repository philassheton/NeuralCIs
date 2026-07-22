import tensorflow as tf
import numpy as np
from keras import backend
from copy import deepcopy

from typing import List, Optional


class _ReduceLROnPlateauTrackBest(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(
            self,
            tensors_to_track: List[tf.Tensor],
            learning_rate_initial: float,
            factor: float,
            absolute_loss_increase_tol: Optional[float] = None,
            relative_loss_increase_tol: Optional[float] = None,
            *args,
            **kwargs,
    ):

        # TODO: At present this is a bit ugly.  It allows the superclass to
        #       update the learning rate, but then might overwrite it with
        #       an brand new optimiser.
        super().__init__(factor=factor, *args, **kwargs)

        self.tensors_to_track = tensors_to_track
        self.tensor_copies = [deepcopy(t) for t in self.tensors_to_track]
        self.learning_rate_initial = learning_rate_initial
        self.factor = factor

        if relative_loss_increase_tol is None:
            self.tolerate_abs_not_rel_increase = True
            if absolute_loss_increase_tol is not None:
                self.loss_increase_tol = absolute_loss_increase_tol
            else:
                self.loss_increase_tol = 0.
        else:
            self.tolerate_abs_not_rel_increase = False
            if absolute_loss_increase_tol is not None:
                raise Exception("Cannot have both absolute AND relative "
                                "loss tolerances!")
            else:
                self.loss_increase_tol = relative_loss_increase_tol

    def backup_values(self):
        for tensor, copy in zip(self.tensors_to_track, self.tensor_copies):
            copy.assign(tensor)

    def restore_best(self):
        for tensor, copy in zip(self.tensors_to_track, self.tensor_copies):
            tensor.assign(copy)

    def initialize_optimizer(
            self,
            learning_rate: float,
    ) -> None:

        self.optimizer_config["learning_rate"] = learning_rate
        self.model.optimizer = self.optimizer_class.from_config(
            self.optimizer_config
        )

    def on_train_begin(
            self,
            logs=None
    ):

        super().on_train_begin(logs)
        self.optimizer_class = self.model.optimizer.__class__
        self.optimizer_config = self.model.optimizer.get_config()
        self.initialize_optimizer(self.learning_rate_initial)

    def current_loss(self, logs):
        return logs.get(self.monitor)

    def on_epoch_end(
            self,
            epoch: int,
            logs=None,
    ) -> None:

        current_loss = self.current_loss(logs)
        best_before_call = self.best if self.best is not None else float("inf")
        wait_before_call = self.wait

        if epoch is not None:
            super().on_epoch_end(epoch, logs)

        if self.in_cooldown():
            return

        if self.monitor_op(current_loss, best_before_call):
            self.backup_values()
        elif wait_before_call + 1 >= self.patience:
            self.restore_if_increase_intolerable(current_loss,
                                                 wipe_momentum_on_restore=True)

    def restore_if_increase_intolerable(
            self,
            current_loss: float,
            wipe_momentum_on_restore: bool = False,
    ):

        if self.tolerate_abs_not_rel_increase:
            increase_vs_best = current_loss - self.best
        else:
            increase_vs_best = current_loss / self.best

        loss_up_too_much = increase_vs_best > self.loss_increase_tol

        if loss_up_too_much:
            tf.print(f"Restoring old ({self.best} beats {current_loss})."
                     f"  Increase: {increase_vs_best},"
                     f"  vs tolerable: {self.loss_increase_tol}")
            self.restore_best()
            if wipe_momentum_on_restore:
                opt = self.model.optimizer
                lr = float(tf.convert_to_tensor(opt.learning_rate).numpy())
                self.initialize_optimizer(lr)

        else:
            tf.print(f"Tolerating this increase."
                     f"  Increase: {increase_vs_best},"
                     f"  vs tolerable: {self.loss_increase_tol}")

    def on_train_end(
            self,
            logs=None,
    ):

        current_loss = self.current_loss(logs)
        self.restore_if_increase_intolerable(current_loss)
