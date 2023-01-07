import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore

from typing import Tuple, Union
from tensor_annotations.tensorflow import Tensor1, Tensor2
from tensor_annotations.tensorflow import float32 as tf32
from neuralcis.common import Samples
import tensor_annotations.tensorflow as ttf


@tf.function
def samples_size_to_std_uniform(
        n: Tensor1[tf32, Samples],
        min_n: ttf.float32,
        max_n: ttf.float32
) -> Tensor1[tf32, Samples]:

    return uniform_to_std_uniform(
        tf.math.log(n),
        tf.math.log(min_n),
        tf.math.log(max_n)
    )


@tf.function
def samples_size_from_std_uniform(
        std_uniform: Tensor1[tf32, Samples],
        min_n: ttf.float32,
        max_n: ttf.float32
) -> Tensor1[tf32, Samples]:

    log_min = tf.math.log(min_n)
    log_max = tf.math.log(max_n)
    log_n = uniform_from_std_uniform(std_uniform, log_min, log_max)
    return tf.math.exp(log_n)


@tf.function
def uniform_from_std_uniform(
        std_uniform: Union[Tensor1, Tensor2],
        min_u: ttf.float32,
        max_u: ttf.float32
) -> Union[Tensor1, Tensor2]:

    return std_uniform * (max_u - min_u) + min_u                               # type: ignore


@tf.function
def uniform_to_std_uniform(
        uniform: Union[Tensor1, Tensor2],
        min_u: ttf.float32,
        max_u: ttf.float32
) -> Union[Tensor1, Tensor2]:

    return (uniform - min_u) / (max_u - min_u)                                 # type: ignore


@tf.function(input_signature=(tf.TensorSpec([None], tf.float32),
                              tf.TensorSpec([None], tf.float32),
                              tf.TensorSpec([None], tf.float32)))
def generate_group_statistics(
        n_in_group: Tensor1[tf32, Samples],
        mus: Tensor1[tf32, Samples],
        sigmas: Tensor1[tf32, Samples]
) -> Tuple[Tensor1[tf32, Samples], Tensor1[tf32, Samples]]:
    df = (n_in_group - 1.)

    ses = sigmas / tf.math.sqrt(n_in_group)
    ms = tf.random.normal(tf.shape(sigmas), mus, ses)
    variances = (
            tfp.distributions.Chi2(df).sample() / df * tf.square(sigmas)
    )

    return ms, variances
