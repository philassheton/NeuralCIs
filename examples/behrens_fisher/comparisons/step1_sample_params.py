# The goal here is to generate a grid of params to test our methods against.
# In order to make sure we have nice clean sampling distributions, we will
# avoid boundary cases that can generate degenerate samples (e.g. all values
# zero or one in the binary variable).

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import behfish_analyse_funcs as behfish

from tensor_annotations.tensorflow import Tensor1
from tensor_annotations.tensorflow import float32 as tf32

import typing
from tensor_annotations import axes
Samples = typing.NewType("Samples", axes.Axis)


NUM_PARAM_SAMPLES = 10000
MAX_TARGET_POWER = 0.90
ALPHA = 0.05

PARAM_SEEDS = {
    'MUDIFF': 0,
    'SIGMA1': 1,
    'SIGMA2': 2,
    'N1': 3,
    'N2': 4,
    'UP_DOWN': 5,
    'TARGET_POWER': 6,
}
SECONDARY_SEED = 42


def __sample_uniform(
        number_to_draw: int,
        param_name: str,
        minval: float,
        maxval: float,
) -> Tensor1[tf32, Samples]:

    param_seed = PARAM_SEEDS[param_name]
    seed = tf.constant([SECONDARY_SEED, param_seed], dtype=tf.int32)
    return tf.random.stateless_uniform((number_to_draw,), seed, minval, maxval)


def __sample_log_uniform(
        number_to_draw: int,
        param_name: str,
        minval: float,
        maxval: float,
) -> Tensor1[tf32, Samples]:

    minval_log = np.log(minval)
    maxval_log = np.log(maxval)
    uniform = __sample_uniform(number_to_draw,
                               param_name,
                               minval_log,
                               maxval_log)
    return tf.math.exp(uniform)


def mudiff_offset_for_power(
        sigma1,
        sigma2,
        n1,
        n2,
        target_powers,
        alpha: float,
):
    v = tf.square(sigma1) / n1 + tf.square(sigma2) / n2
    s = tf.sqrt(v)
    z_alpha = tfp.distributions.Normal(0., 1.).quantile(1 - alpha/2.)
    z_beta = tfp.distributions.Normal(0., 1.).quantile(target_powers)
    return s * (z_alpha + z_beta)


def sample_n_params(
        num_param: int,
) -> dict[str, Tensor1[tf32, Samples]]:

    n1 = tf.floor(__sample_log_uniform(num_param, "N1", 3., 101.))
    n2 = tf.floor(__sample_log_uniform(num_param, "N2", 3., 101.))

    sigma1 = __sample_log_uniform(num_param, "SIGMA1", 0.01, 100.)
    sigma2 = __sample_log_uniform(num_param, "SIGMA2", 0.333, 3.) * sigma1
    mudiff = __sample_uniform(num_param, "MUDIFF", -100., 100.) * sigma1
    target_powers = __sample_uniform(num_param, "TARGET_POWER",
                                     ALPHA, MAX_TARGET_POWER)

    mudiff_power_offsets = mudiff_offset_for_power(sigma1, sigma2, n1, n2,
                                                   target_powers, ALPHA)
    mudiff_power_directions = tf.where(
        __sample_uniform(num_param, "UP_DOWN", 0., 1.) > 0.5,
        1.,
        -1.,
    )
    mudiff_power = mudiff + mudiff_power_offsets*mudiff_power_directions

    return {
        "mudiff": mudiff,
        "mudiff_power": mudiff_power,
        "target_power": target_powers,

        "sigma1": sigma1,
        "sigma2": sigma2,
        "n1": n1,
        "n2": n2,
    }


params = sample_n_params(NUM_PARAM_SAMPLES)
behfish.save_params_dict(params)
