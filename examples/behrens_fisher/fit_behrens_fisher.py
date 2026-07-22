import os
import tensorflow as tf

import neuralcis
from neuralcis import Stat, Param, KnownParam
from neuralcis import Location, Scale, PositiveCount
import tensorflow_probability as tfp


def sampling_distribution_fn(
        mudiff, sigma1, sigma2,  # Unknown params
        n1, n2,                  # Known params
):

    df1 = n1 - 1.
    df2 = n2 - 1.

    z1 = tf.random.normal(tf.shape(mudiff))
    z2 = tf.random.normal(tf.shape(mudiff))
    chi_sq1 = tfp.distributions.Chi2(df1).sample(1)[0, :]
    chi_sq2 = tfp.distributions.Chi2(df2).sample(1)[0, :]

    mu1_hat = z1 * sigma1 / tf.math.sqrt(n1)
    mu2_hat = z2 * sigma2 / tf.math.sqrt(n2) + mudiff
    mudiff_hat = mu2_hat - mu1_hat

    sigma1_hat = sigma1 * tf.math.sqrt(chi_sq1 / df1)
    sigma2_hat = sigma2 * tf.math.sqrt(chi_sq2 / df2)

    return {"mudiff_hat": mudiff_hat,
            "sigma1_hat": sigma1_hat,
            "sigma2_hat": sigma2_hat}


def interest_fn(
        mudiff, sigma1, sigma2,  # Unknown params
        n1, n2,                  # Known params
):

    return mudiff


def estimates_fn(
        mudiff_hat, sigma1_hat, sigma2_hat,  # Statistics
        n1, n2,                              # Known parameters
):

    return {"mudiff": mudiff_hat, "sigma1": sigma1_hat, "sigma2": sigma2_hat}


def transform_on_stats_fn(
        mudiff_hat, sigma1_hat, sigma2_hat,  # Statistics
        mudiff, sigma1, sigma2,              # Unknown parameters
        n1, n2,                              # Known parameters
):

    return {"sigma2_1_ratio_hat": sigma2_hat / sigma1_hat,

            "mudiff": (mudiff - mudiff_hat) / sigma1_hat,  # Transformed params
            "sigma1": sigma1 / sigma1_hat,
            "sigma2": sigma2 / sigma1_hat,

            "n1": n1,
            "n2": n2}


cis = neuralcis.NeuralCIs(
    sampling_distribution_fn,
    interest_fn,
    estimates_fn,
    ["mudiff", "sigma1", "sigma2"],
    ["mudiff_hat", "sigma1_hat", "sigma2_hat"],
    ["n1", "n2"],

    transform_on_stats_fn,
    ["sigma2_1_ratio_hat"],

    mudiff=Param(Location((-3., 3.)), (0., 0.)),
    sigma1=Param(Scale((0.333, 3.)), (1., 1.)),
    sigma2=Param(Scale((0.1, 10.)), (0.333, 3.)),

    n1=KnownParam(PositiveCount((3., 100.))),
    n2=KnownParam(PositiveCount((3., 100.))),

    mudiff_hat=Stat(Location((-3., 3.))),
    sigma1_hat=Stat(Scale((.3, 3.))),
    sigma2_hat=Stat(Scale((.1, 10.))),

    sigma2_1_ratio_hat=Stat(Scale((.3, 3.))),

    train_initial_weights=False,
)
cis.fit()
cis.save('saved_model')
