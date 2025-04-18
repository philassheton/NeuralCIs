# MIT License
#
# Copyright (c) 2024 Szymon Miłosz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer


class AdamWScheduleFree(Optimizer):
    r"""
    A TensorFlow/Keras implementation of the Schedule-Free AdamW optimizer as described in
    [Defazio et al., 2024](https://arxiv.org/abs/2405.15682).

    This implementation does not rely on built-in learning rate scheduling; instead, it uses
    a `warmup_steps` parameter for linear warmup. In addition, two custom methods,
    `set_in_eval_mode()` and `set_in_train_mode()`, are provided to adjust model variables
    using stored “shadow” variables (slots) for iterate averaging.

    **Important:** In this version, you must explicitly call the optimizer’s `build()`
    method with all model trainable variables after the model has built its weights and
    before training begins.

    Args:
        learning_rate: A numeric value or Tensor, the initial learning rate. Defaults to 0.0025.
        beta_1: Float. The momentum parameter. Defaults to 0.9.
        beta_2: Float. The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
        epsilon: Float. A small constant for numerical stability. Defaults to 1e-8.
        weight_decay: Float. The weight decay (L2 penalty) term. Defaults to 0.0.
        warmup_steps: Int. The number of steps for linear learning rate warmup. Defaults to 0.
        r: Float. The exponent used in the weight update computation. Defaults to 0.0.
        weight_lr_power: Float. The power used to weight the learning rate maximum during warmup. Defaults to 2.0.
        clipnorm: (Optional) clipping norm.
        clipvalue: (Optional) clipping value.
        global_clipnorm: (Optional) global clipping norm.
        name: String. The optimizer’s name. Defaults to "SFAdamW".
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            learning_rate=0.0025,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            weight_decay=0.0,
            warmup_steps=0,
            r=0.0,
            weight_lr_power=2.0,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            name="SFAdamW",
            **kwargs,
    ):
        # Pass only the name (and other recognized kwargs) to the base class.
        super().__init__(name, **kwargs)

        # Use the built-in hyperparameter mechanism.
        self._set_hyper("learning_rate", learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.r = r
        self.weight_lr_power = weight_lr_power

        # Optional: store clipping settings (if you wish to use them manually).
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.global_clipnorm = global_clipnorm

        # Variables to track the maximum learning rate and cumulative weight.
        self.weight_sum = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.lr_max = tf.Variable(-1.0, dtype=tf.float32, trainable=False)

        # A flag to indicate train vs. eval mode.
        self.train_mode = tf.Variable(True, dtype=tf.bool, trainable=False)

        # Placeholders for slot variables and a lookup dictionary.
        # These will be created in the build() method.
        self._z = None  # “Shadow” variables for parameter averaging.
        self._v = None  # Accumulators for the second moment.
        self._index_dict = None  # Maps variable names to their slot index.

    def add_variable_from_reference(
            self,
            model_variable,
            variable_name,
            initial_value=None
    ):
        """
        Helper to create a slot variable for a given model variable.
        For the "z" slot, the initial value is typically the variable's current value.
        """
        if initial_value is None:
            initial_value = model_variable
        return self.add_slot(model_variable, variable_name,
                             initializer=initial_value)

    def _var_key(
            self,
            var
    ):
        """
        Returns a unique key for each variable. Here we use the variable’s name,
        which is stable once built.
        """
        return var.name

    def build(
            self,
            var_list
    ):
        """
        Initialize the slot variables. This method must be called once, after your model
        has built all its weights.

        Args:
            var_list: List of all model trainable variables.
        """
        # If _index_dict is already built, we assume the slots are already set up.
        if self._index_dict is not None:
            return

        self._z = []
        self._v = []
        self._index_dict = {}
        for var in var_list:
            key = self._var_key(var)
            # For z, we initialize with the variable's current value.
            self._z.append(
                self.add_variable_from_reference(var, "z", initial_value=var))
            # For v, use zeros (same shape as var).
            self._v.append(
                self.add_slot(var, "v", initializer=tf.zeros_like(var)))
            self._index_dict[key] = len(self._z) - 1

    def update_step(
            self,
            gradient,
            variable
    ):
        """
        Update a single variable using its gradient.
        This follows the Schedule-Free AdamW update equations.
        """
        # Retrieve the learning rate as a tensor.
        lr = tf.cast(self._get_hyper("learning_rate"), variable.dtype)
        gradient = tf.cast(gradient, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        warmup_steps = tf.cast(self.warmup_steps, variable.dtype)

        # Apply linear warmup if needed.
        schedule = tf.cond(tf.greater_equal(warmup_steps, local_step),
                           lambda: local_step / warmup_steps,
                           lambda: 1.0)

        # Bias correction for the second moment.
        bias_correction2 = 1 - tf.pow(tf.cast(self.beta_2, variable.dtype),
                                      local_step)
        lr = lr * schedule
        lr = tf.multiply(tf.sqrt(bias_correction2), lr)

        # Update the maximum observed learning rate.
        self.lr_max.assign(tf.maximum(self.lr_max, lr))

        # Compute a weight for aggregate updates.
        weight = tf.multiply(tf.pow(local_step, self.r),
                             tf.pow(self.lr_max, self.weight_lr_power))
        self.weight_sum.assign_add(weight)
        ckp1 = tf.math.divide_no_nan(weight, self.weight_sum)

        # Retrieve the corresponding slot variables.
        var_key = self._var_key(variable)
        y = variable  # The "current" parameter value.
        z = self._z[self._index_dict[var_key]]
        v = self._v[self._index_dict[var_key]]

        # Update the second moment accumulator.
        v.assign(tf.add(tf.multiply(v, self.beta_2),
                        tf.multiply(1 - self.beta_2, tf.square(gradient))))
        denominator = tf.add(tf.sqrt(v), self.epsilon)
        gradient_normalized = tf.divide(gradient, denominator)

        # Optionally apply weight decay.
        if self.weight_decay > 0:
            if self._use_weight_decay(variable):
                gradient_normalized = gradient_normalized + tf.multiply(y,
                                                                        self.weight_decay)

        # Update the variable using the shadow variable.
        y.assign_add(tf.multiply(ckp1, tf.subtract(z, y)))
        alpha = tf.multiply(lr, tf.multiply(self.beta_1, (1 - ckp1)) - 1)
        y.assign_add(tf.multiply(alpha, gradient_normalized))
        z.assign_sub(tf.multiply(gradient_normalized, lr))

    def _use_weight_decay(
            self,
            var
    ):
        """
        Decide whether to apply weight decay to a variable.
        Modify this method if you want to skip weight decay for certain parameters,
        for example biases or normalization parameters.
        """
        return True

    def set_in_eval_mode(
            self,
            var_list
    ):
        """
        Adjust model variables to evaluation mode using parameter averaging.
        This applies an adjustment based on the stored shadow variables.
        """
        if self.train_mode:
            weight = 1 - 1 / self.beta_1
            for var in var_list:
                var_key = self._var_key(var)
                idx = self._index_dict[var_key]
                z = self._z[idx]
                var.assign_add(tf.multiply(weight, tf.subtract(z, var)))
            self.train_mode.assign(False)

    def set_in_train_mode(
            self,
            var_list
    ):
        """
        Restore model variables to training mode from evaluation mode.
        """
        if not self.train_mode:
            weight = 1 - self.beta_1
            for var in var_list:
                var_key = self._var_key(var)
                idx = self._index_dict[var_key]
                z = self._z[idx]
                var.assign_add(tf.multiply(weight, tf.subtract(z, var)))
            self.train_mode.assign(True)

    # -- Required methods for tf.keras.optimizers.Optimizer integration --

    def _resource_apply_dense(
            self,
            grad,
            var,
            apply_state=None
    ):
        """
        Dense gradient update. Assumes that build() has already been called.
        """
        self.update_step(grad, var)
        return tf.no_op()

    def _resource_apply_sparse(
            self,
            grad,
            var,
            indices,
            apply_state=None
    ):
        """
        Sparse gradient update. Converts sparse gradients to dense form.
        """
        grad_dense = tf.convert_to_tensor(grad)
        self.update_step(grad_dense, var)
        return tf.no_op()

    def _apply_weight_decay(
            self,
            variables
    ):
        """
        This method is left blank because weight decay is applied inside update_step.
        """
        pass


class ChangeModeCallback(tf.keras.callbacks.Callback):
    def __init__(
            self,
            steps_per_epoch
    ):
        super(ChangeModeCallback, self).__init__()
        self.steps_per_epoch = steps_per_epoch

    def on_epoch_begin(
            self,
            epoch,
            logs=None
    ):
        self.model.optimizer.set_in_train_mode(self.model.trainable_variables)

    def on_train_batch_end(
            self,
            batch,
            logs=None
    ):
        if batch == self.steps_per_epoch - 1:
            self.model.optimizer.set_in_eval_mode(
                self.model.trainable_variables)
