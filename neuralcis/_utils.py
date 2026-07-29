import tensorflow as tf

from . import common
from .common import Samples, UnknownParams, KnownParams, Params
from typing import Tuple
from tensor_annotations.tensorflow import Tensor2, float32 as tf32


def _soft_floor_at_zero(
        values: tf.Tensor,
        soft_floor_ceiling: float = common.SOFT_FLOOR_CEILING,
) -> tf.Tensor:

    # TODO: Consider whether this needs to adapt the scale of the variables
    #       used (I think not, now that all variables are in [-1, 1].)
    punitive_but_not_zero_values = soft_floor_ceiling * tf.math.sigmoid(values)
    values_floored = tf.math.maximum(values, punitive_but_not_zero_values)
    return values_floored


def concat_unknown_and_known_params(
        unknown_params: Tensor2[tf32, Samples, UnknownParams],
        known_params: Tensor2[tf32, Samples, KnownParams],
) -> Tensor2[tf32, Samples, Params]:

    return tf.concat([unknown_params, known_params], axis=1)


def split_unknown_and_known_params(
        params: Tensor2[tf32, Samples, Params],
        num_unknown_param: int,
        num_known_param: int,
) -> Tuple[Tensor2[tf32, Samples, UnknownParams],
           Tensor2[tf32, Samples, KnownParams]]:

    assert params.shape[1] == num_unknown_param + num_known_param
    return tf.split(params, [num_unknown_param, num_known_param], axis=1)


def known_params_from_params(
        params: Tensor2[tf32, Samples, Params],
        num_unknown_param: int,
        num_known_param: int,  # Force entering both to avoid muddling them up
) -> Tensor2[tf32, Samples, KnownParams]:

    assert params.shape[1] == num_unknown_param + num_known_param
    return params[:, num_unknown_param:]
