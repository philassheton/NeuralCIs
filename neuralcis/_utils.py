import tensorflow as tf

from neuralcis import common


@tf.function
def _soft_floor_at_zero(
        values: tf.Tensor,
        soft_floor_ceiling: float = common.SOFT_FLOOR_CEILING,
) -> tf.Tensor:

    # TODO: Consider whether this needs to adapt the scale of the variables
    #       used (I think not, now that all variables are in [-1, 1].)
    punitive_but_not_zero_values = soft_floor_ceiling * tf.math.sigmoid(values)
    values_floored = tf.math.maximum(values, punitive_but_not_zero_values)
    return values_floored
