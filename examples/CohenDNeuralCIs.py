from neuralcis.neuralcis import NeuralCIs
from neuralcis import sampling

import tensorflow as tf


D_MIN = tf.constant(-2.)
D_MAX = tf.constant(2.)
N_MIN = tf.constant(3.)
N_MAX = tf.constant(300.)

class CohenDNeuralCIs(NeuralCIs):
    num_good_param = 1
    num_known_param = 2

    def simulate_sampling_distribution(self, pop_d, n1, n2):
        mu2 = pop_d
        mu1 = pop_d * 0.
        sigma1 = sigma2 = pop_d * 0. + 1.

        m1, v1 = sampling.generate_group_statistics(n1, mu1, sigma1)
        m2, v2 = sampling.generate_group_statistics(n2, mu2, sigma2)

        v_pooled = (v1 * (n1 - 1.) + v2 * (n2 - 1.)) / (n1 + n2 - 2.)

        d = (m2 - m1) / tf.math.sqrt(v_pooled)

        return (d, )

    def params_from_std_uniform(self, pop_d, n1, n2):
        return (
            sampling.uniform_from_std_uniform(pop_d, D_MIN, D_MAX),
            sampling.samples_size_from_std_uniform(n1, N_MIN, N_MAX),
            sampling.samples_size_from_std_uniform(n2, N_MIN, N_MAX)
        )

    def params_to_std_uniform(self, pop_d, n1, n2):
        return (
            sampling.uniform_to_std_uniform(pop_d, D_MIN, D_MAX),
            sampling.samples_size_to_std_uniform(n1, N_MIN, N_MAX),
            sampling.samples_size_to_std_uniform(n2, N_MIN, N_MAX)
        )

    def estimates_to_std_uniform(self, d):
        return (
            sampling.uniform_to_std_uniform(d, D_MIN, D_MAX),
        )
