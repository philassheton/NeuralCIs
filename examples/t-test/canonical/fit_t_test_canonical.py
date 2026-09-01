import tensorflow as tf
import tensorflow_probability as tfp

import neuralcis as nci
from neuralcis import Location, Scale, PositiveCount


def sampling_distribution_fn(mu, sigma, n):
    df = n - 1.
    z = tf.random.normal(tf.shape(mu))
    chi_sq = tfp.distributions.Chi2(df).sample(1)[0, :]

    mu_hat = z * sigma / tf.math.sqrt(n) + mu
    sigma_hat = sigma * tf.math.sqrt(chi_sq / df)

    return {"mu_hat": mu_hat, "sigma_hat": sigma_hat}


def interest_fn(mu, sigma, n):
    return mu


def estimates_fn(mu_hat, sigma_hat, n):
    return {"mu": mu_hat, "sigma": sigma_hat}


tf.keras.utils.set_random_seed(12345)
cis = nci.NeuralCIs(
    sampling_distribution_fn,
    interest_fn,
    estimates_fn,

    mu=nci.Param(Location(), (0., 0.), "(mu - mu_hat) / sigma_hat"),
    sigma=nci.Param(Scale(), (1., 1.), "sigma / sigma_hat"),

    n=nci.KnownParam(PositiveCount(), (3., 100.), "n"),

    interest=nci.Interest(Location(), "(interest - mu_hat) / sigma_hat"),

    mu_hat=nci.Stat(Location()),
    sigma_hat=nci.Stat(Scale()),
)
cis.fit()
cis.save("saved_model")
