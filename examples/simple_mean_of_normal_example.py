import tensorflow as tf
from neuralcis import NeuralCIs, Uniform, LogUniform

def normal_sampling_fn(mu, sigma, n):
    std_normal = tf.random.normal(tf.shape(mu))
    mu_hat = std_normal * sigma / tf.math.sqrt(n) + mu
    return {"mu": mu_hat}


cis = NeuralCIs(
    normal_sampling_fn,
    mu=Uniform(-2., 2.),
    sigma=LogUniform(.1, 10.),
    n=LogUniform(3., 300.)
)

cis.fit()
print(cis.p_and_ci(1.96, mu=0., sigma=4., n=16.))