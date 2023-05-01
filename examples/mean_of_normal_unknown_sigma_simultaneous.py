import tensorflow as tf
import tensorflow_probability as tfp
from neuralcis import NeuralCIs, Uniform, LogUniform

def normal_sampling_fn(mu, sigma, n):
    df = n - 1.
    std_normal = tf.random.normal(tf.shape(mu))
    mu_hat = std_normal * sigma / tf.math.sqrt(n) + mu
    sigma_hat = sigma * tfp.distributions.Chi2(df).sample(1) / df
    return {"mu": mu_hat, "sigma": sigma_hat[0, :]}


cis = NeuralCIs(
    normal_sampling_fn,
    mu=Uniform(-2., 2.),
    sigma=LogUniform(.1, 10.),
    n=LogUniform(3., 300.)
)

cis.fit()

# should be p = 0.05, lower bound going to zero
print(cis.p_and_ci(
    {'mu': 0.173, 'sigma': 1.},
    {'mu': 0., 'sigma': 1.1, 'n': 200.}
))