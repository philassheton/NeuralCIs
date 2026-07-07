import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
tf.config.optimizer.set_jit(True)  # Enable XLA globally

import neuralcis
from neuralcis import Location, Scale, SampleSize
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


def contrast_fn(
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
    contrast_fn,
    estimates_fn,
    ["mudiff", "sigma1", "sigma2"],
    ["mudiff_hat", "sigma1_hat", "sigma2_hat"],
    ["n1", "n2"],

    transform_on_stats_fn,
    ["sigma2_1_ratio_hat"],

    mudiff=Location(-3., 3., 0., 0.),
    sigma1=Scale(.3, 3., 1., 1.),
    sigma2=Scale(.1, 10., 0.333, 3.),
    n1=SampleSize(3, 100),
    n2=SampleSize(3, 100),

    mudiff_hat=Location(-3., 3.),
    sigma1_hat=Scale(.3, 3.),
    sigma2_hat=Scale(.1, 10.),

    sigma2_1_ratio_hat=Scale(.3, 3.),

    train_initial_weights=False,
)
cis.fit()
cis.save('saved_model')
