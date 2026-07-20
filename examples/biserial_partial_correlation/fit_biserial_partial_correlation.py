import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
tf.config.optimizer.set_jit(True)  # TODO: selectively turn on XLA instead

import neuralcis
from neuralcis import Correlation, Proportion, SampleSize
import tensorflow_probability as tfp


N_MIN = 10
N_MAX = 100


def sampling_distribution_fn(
        rho_ab_partial,
        rho_ac,
        rho_bc,
        prop_a,
        n,
):

    batch_size = tf.shape(n)[0]
    rho_ab = (rho_ac * rho_bc
              + rho_ab_partial * tf.sqrt((1. - tf.square(rho_ac)) *
                                         (1. - tf.square(rho_bc))))

    z_threshold_a = -tfp.distributions.Normal(0., 1.).quantile(prop_a)

    one = tf.ones_like(rho_ab)
    correlation_matrix = tf.stack([
        tf.stack([one, rho_ab, rho_ac], axis=1),
        tf.stack([rho_ab, one, rho_bc], axis=1),
        tf.stack([rho_ac, rho_bc, one], axis=1),
    ], axis=2)

    cholesky = tf.linalg.cholesky(correlation_matrix)

    z = tf.random.normal((batch_size, 3, N_MAX))
    z_correlated = tf.linalg.matmul(cholesky, z)

    a, b, c = tf.split(z_correlated, 3, axis=1)
    a = tf.cast(a > z_threshold_a[:, None, None], tf.float32)
    X = tf.stack([a[:, 0, :], b[:, 0, :], c[:, 0, :]], axis=2)

    mask = tf.cast(tf.sequence_mask(n, N_MAX)[:, :, None], tf.float32)
    X_mean = tf.reduce_sum(X*mask, axis=1, keepdims=True) / n[:, None, None]
    X = (X - X_mean) * mask

    X_sum_pairwise = tf.reduce_sum(X * tf.gather(X, [1, 2, 0], axis=2), axis=1)
    X_sum_sq = tf.reduce_sum(tf.square(X), axis=1)
    X_sum_sq_pairwise = X_sum_sq * tf.gather(X_sum_sq, [1, 2, 0], axis=1)

    correlations_hat = X_sum_pairwise / tf.sqrt(X_sum_sq_pairwise + 1e-10)

    return {
        'rho_ab_hat': correlations_hat[:, 0],
        'rho_bc_hat': correlations_hat[:, 1],
        'rho_ac_hat': correlations_hat[:, 2],
        'prop_a_hat': X_mean[:, 0, 0],  # Make prop_a a known param for now!
    }


def interest_fn(
        rho_ab_partial,
        rho_ac,
        rho_bc,
        prop_a,
        n,
):

    return rho_ab_partial


def estimates_fn(
        rho_ab_hat, rho_ac_hat, rho_bc_hat, prop_a_hat,
        n,
):

    # TODO: This hack is needed to make sure we stay the right side of stuff;
    #       try to fix this problem in the main sim so this is not needed.
    prop_a_hat = prop_a_hat * 0.9998 + 0.0001
    rho_ab_hat = rho_ab_hat * 0.9998 + 0.0001
    rho_bc_hat = rho_bc_hat * 0.9998 + 0.0001
    rho_ac_hat = rho_ac_hat * 0.9998 + 0.0001

    std_normal = tfp.distributions.Normal(0., 1.)
    threshold_a = std_normal.quantile(1. - prop_a_hat)
    prob_threshold = std_normal.prob(threshold_a)

    biserial_correction = (tf.sqrt(prop_a_hat * (1. - prop_a_hat))
                           / prob_threshold)

    # This is an approximation that keeps our estimate within bounds.  Instead
    #   of multiplying rho_ab_hat by the correction (which is the "correct"
    #   approach), atanh transform before multiplying to "fade out" the
    #   correction as we get to higher rho_ab_hat.  This avoids getting rho_ab
    #   out of bounds.

    # An alternative might be to just simply use the point biserial
    #   correlation, but this is then unlikely to push us far enough to
    #   (-1, +1).

    rho_ab = tf.tanh(biserial_correction * tf.atanh(rho_ab_hat))
    rho_ac = tf.tanh(biserial_correction * tf.atanh(rho_ac_hat))
    rho_bc = rho_bc_hat

    rho_ab_partial = ((rho_ab - rho_ac * rho_bc)
                      / tf.sqrt((1. - rho_ac**2) * (1. - rho_bc**2)))

    return {'rho_ab_partial': rho_ab_partial,
            'rho_ac': rho_ac,
            'rho_bc': rho_bc,
            'prop_a': prop_a_hat}


cis = neuralcis.NeuralCIs(
    sampling_distribution_fn,
    interest_fn,
    estimates_fn,
    ["rho_ab_partial", "rho_bc", "rho_ac", "prop_a"],
    ["rho_ab_hat", "rho_bc_hat", "rho_ac_hat", "prop_a_hat"],
    ["n"],

    None, None,

    rho_ab_partial=Correlation(),
    rho_ac=Correlation(),
    rho_bc=Correlation(),

    rho_ab_hat=Correlation(),
    rho_ac_hat=Correlation(),
    rho_bc_hat=Correlation(),
    prop_a_hat=Proportion(0.05, 0.95),

    prop_a=Proportion(0.05, 0.95),
    n=SampleSize(N_MIN, N_MAX),

    param_sampling_regularize_jitter_multiply=0.1,
    param_sampling_regularize_jitter_add=0.03,

    train_initial_weights=False,
)

cis.fit()
cis.save('saved_model', 'full')
