import tensorflow as tf
import tensorflow_probability as tfp
from neuralcis import NeuralCIs, Uniform, LogUniform


def normal_sampling_fn(mu, sigma, n):
    df = n - 1.
    std_normal = tf.random.normal(tf.shape(mu))
    chi_sq = tfp.distributions.Chi2(df).sample(1)

    mu_hat = std_normal * sigma / tf.math.sqrt(n) + mu
    sigma_hat = sigma * tf.math.sqrt(chi_sq / df)

    return {"mu_hat": mu_hat, "sigma_hat": sigma_hat[0, :]}


def contrast_fn(mu, sigma, n):
    return mu


def transform_on_params_fn(mu_hat, sigma_hat, mu, sigma, n):
    return {
        "mu_hat": (mu_hat - mu) / sigma,
        "sigma_hat": sigma_hat / sigma,
        "n": n,
    }


def transform_on_estimates_fn(mu_hat, sigma_hat, mu, sigma, n):
    return {
        # TODO -- make it so the 0. and 1. below are automated and linked
        #         to the distribution setup
        "mu_hat": 0.,
        "sigma_hat": 1.,
        "mu": (mu - mu_hat) / sigma_hat,
        "sigma": sigma / sigma_hat,
        "n": n,
    }


cis = NeuralCIs(
    normal_sampling_fn,
    contrast_fn,
    transform_on_params_fn,
    transform_on_estimates_fn,
    mu=Uniform(-4., 4., 0., 0.),
    sigma=LogUniform(.1, 10., 1., 1.),
    n=LogUniform(3., 300.),
)


cis.fit()


# should be p = 0.05, lower bound going to zero
print(cis.p_and_ci(
    {'mu': 0.139, 'sigma': 1.},
    {'mu': 0., 'sigma': 1., 'n': 200.},
))

