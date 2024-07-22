import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore

from typing import Tuple, Union
from tensor_annotations.tensorflow import Tensor1, Tensor2
from tensor_annotations.tensorflow import float32 as tf32
from neuralcis.common import Samples
import tensor_annotations.tensorflow as ttf


@tf.function
def samples_size_from_std_uniform(
        std_uniform: Tensor1[tf32, Samples],
        min_n: ttf.float32,
        max_n: ttf.float32,
) -> Tensor1[tf32, Samples]:

    """Transform a standard uniform variable into a sample size.

    The mapping is such that the log transform of the resulting sample size
    will be uniformly distributed (thereby placing more emphasis on smaller
    sample sizes).

    NOTE that the sample sizes generated by this function are floating
    point.  The other functions in this package can deal with fractional
    sample sizes.  Using fractional samples sizes smooths our sample space,
    so it is good to when possible, but in other situations, these numbers
    might need rounding.

    :param std_uniform: A 1D `Tensor` of `tf.float32`s, random, standard
        uniform.
    :param min_n: `tf.float32`, smallest sample size that can be sampled.
    :param max_n: `tf.float32`, largest sample size that can be sampled.
    :return: A 1D `Tensor` of `tf.float32`s, random sample sizes between
        min_n and max_n.
    """

    log_min = tf.math.log(min_n)
    log_max = tf.math.log(max_n)
    log_n = uniform_from_std_uniform(std_uniform, log_min, log_max)
    return tf.math.exp(log_n)


@tf.function
def samples_size_to_std_uniform(
        n: Tensor1[tf32, Samples],
        min_n: ttf.float32,
        max_n: ttf.float32,
) -> Tensor1[tf32, Samples]:

    """Inversion of the mapping in `samples_size_from_std_uniform`.

    :param n: A 1D `Tensor` of `tf.float32`s, representing sample sizes
        between min_n and max_n.
    :param min_n: `tf.float32`, smallest sample size that can be present.
    :param max_n: `tf.float32`, largest sample size that can be present.
    :return: A 1D `Tensor` of `tf.float32`s, std uniform values.
    """

    return uniform_to_std_uniform(
        tf.math.log(n),
        tf.math.log(min_n),
        tf.math.log(max_n),
    )


@tf.function
def uniform_to_std_uniform(
        uniform: Union[Tensor1, Tensor2],
        min_u: ttf.float32,
        max_u: ttf.float32,
) -> Union[Tensor1, Tensor2]:

    """Transform any uniform distributed value into a standard uniform one.

    :param uniform: 1D `Tensor` of `tf.float32`s, uniform in [min_u, max_u].
    :param min_u: `tf.float32`, lower limit of uniform.
    :param max_u: `tf.float32`, upper limit of uniform.
    :return: 1D `Tensor` of `tf.float32`s, uniform in [0, 1].
    """

    return (uniform - min_u) / (max_u - min_u)                                 # type: ignore


@tf.function
def uniform_from_std_uniform(
        std_uniform: Union[Tensor1, Tensor2],
        min_u: ttf.float32,
        max_u: ttf.float32,
) -> Union[Tensor1, Tensor2]:

    """Transform a standard uniform variable into a different uniform.

    :param std_uniform: 1D `Tensor` of `tf.float32`s, uniform in [0, 1]
    :param min_u: `tf.float32`, lower limit of uniform.
    :param max_u: `tf.float32`, upper limit of uniform.
    :return: 1D `Tensor` of `tf.float32`s, uniform in [min_u, max_u].
    """

    return std_uniform * (max_u - min_u) + min_u                               # type: ignore


@tf.function(input_signature=(tf.TensorSpec([None], tf.float32),
                              tf.TensorSpec([None], tf.float32),
                              tf.TensorSpec([None], tf.float32)))
def generate_group_statistics(
        n_in_group: Tensor1[tf32, Samples],
        mus: Tensor1[tf32, Samples],
        sigmas: Tensor1[tf32, Samples],
) -> Tuple[Tensor1[tf32, Samples], Tensor1[tf32, Samples]]:

    """Sample means and variances for samples from normal distributions.

    Input to this function is a set of 1D `Tensor`s, all of the same size.
    These define a series of different normal distributions (one for each
    element in the tensors).  From each of these normal distributions,
    one sample of size `n_in_group` is drawn, and the means and variances
    for each of these groups is returned.

    NOTE that the sample sizes n_in_group are floating point and CAN be
    fractional.

    :param n_in_group: 1D `Tensor` of `tf.float32`s, sample size at each
        sample.
    :param mus: 1D `Tensor` of `tf.float32`s, popn mean at each sample.
    :param sigmas: 1D `Tensor` of `tf.float32`s, popn SD at each sample.
    :return: Tuple of two 1D `Tensor` of `tf.float32`s: (1) one mean sampled
        for each element in the input tensors; (2) one variance sampled for
        each element in the input tensors.
    """

    df = (n_in_group - 1.)

    std_errors = sigmas / tf.math.sqrt(n_in_group)
    means = tf.random.normal(tf.shape(sigmas), mus, std_errors)
    variances = tfp.distributions.Chi2(df).sample() / df * tf.square(sigmas)

    return means, variances
