import biparcorr_analyse_funcs as biparcorr

import tensorflow as tf
import tensorflow_probability as tfp

import functools

# typing
from typing import Optional, Union, Tuple, Dict
from biparcorr_analyse_funcs import Batch, Samples, Stats, UnknownParams, Ys
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3
from tensor_annotations.tensorflow import float32 as tf32, int64 as ti64


MAX_ABS_RHO = 0.99
HYPERPARAMETERS_DEFAULT = {
    'adam_iterations': tf.constant(120, dtype=tf.int64),
    'adam_polish_iterations': tf.constant(40, dtype=tf.int64),
    'learning_rate_multiplier': tf.constant(1.0, dtype=tf.float32),
    'learning_rate_polish_decay': tf.constant(0.9, dtype=tf.float32),
    'beta1': tf.constant(0.8, dtype=tf.float32),
    'beta2': tf.constant(0.99, dtype=tf.float32),
}
UNROLL_WHEN_COMPILING = False  # Faster but much slower to compile


def log_likelihood(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        rho_ab_partial: Tensor1[tf32, Batch],
        rho_bc: Tensor1[tf32, Batch],
        rho_ac: Tensor1[tf32, Batch],
        prop_a: Tensor1[tf32, Batch],
        n: Tensor1[tf32, Batch],
) -> Tensor1[tf32, Batch]:

    # Note: the likelihood methods used here have the advantage that they
    #       magically know the variances of b and c.

    rho_ab = (rho_bc * rho_ac
              + rho_ab_partial * tf.sqrt((1. - tf.square(rho_bc)) *
                                         (1. - tf.square(rho_ac))))

    z_threshold_a = -tfp.distributions.Normal(0., 1.).quantile(prop_a)

    bc = stats_tensor[:, :, 1:]
    one = tf.ones_like(rho_bc)
    cov_bc = tf.stack([
        tf.stack([one, rho_bc], axis=-1),
        tf.stack([rho_bc, one], axis=-1),
    ], axis=-1)
    mvn_bc = tfp.distributions.MultivariateNormalTriL(
        loc=tf.zeros(2),
        scale_tril=tf.linalg.cholesky(cov_bc)[:, None, :, :],
    )
    log_prob_bc = mvn_bc.log_prob(bc)

    a = stats_tensor[:, :, 0]
    cov_a_bc = tf.stack([rho_ab, rho_ac], axis=-1)
    cov_bc_inv = tf.linalg.inv(cov_bc)
    beta_a_given_bc = tf.linalg.matvec(cov_bc_inv, cov_a_bc)
    mean_given_bc = tf.linalg.matvec(bc, beta_a_given_bc)
    var_given_bc = 1. - tf.linalg.matvec(cov_a_bc[:, None, :], beta_a_given_bc)
    sd_given_bc = tf.sqrt(var_given_bc)

    z_score__threshold_given_bc = ((z_threshold_a[:, None] - mean_given_bc)
                                   / sd_given_bc)
    p_one = 1. - 0.5 * (
                1. + tf.math.erf(z_score__threshold_given_bc / tf.sqrt(2.)))
    p_one = tf.clip_by_value(p_one, 1e-7, 1. - 1e-7)

    one = tf.constant(1.)
    log_prob_a = a*tf.math.log(p_one) + (one - a)*tf.math.log(one - p_one)

    log_prob_abc_masked = biparcorr.n_mask(n) * (log_prob_a + log_prob_bc)

    log_likelihood_final = tf.reduce_sum(log_prob_abc_masked, axis=1)

    return log_likelihood_final                                                # type: ignore


def penalise_log_likelihood(
        log_likelihoods: Tensor1[tf32, Batch],
        params_unknown_transformed: Tensor2[tf32, Batch, UnknownParams],
) -> Tensor1[tf32, Batch]:

    max_abs_rho_transformed = transform_correlations(tf.constant(MAX_ABS_RHO))
    max_abs = tf.reduce_max(tf.math.abs(params_unknown_transformed), axis=1)
    penalty_unscaled = tf.nn.relu(max_abs - max_abs_rho_transformed)

    return log_likelihoods - 10.*penalty_unscaled


def log_likelihood_from_transformed_params(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        n: Tensor1[tf32, Batch],
        params_unknown_transformed: Tensor2[tf32, Batch, UnknownParams],
        penalise_boundaries: bool = True,
) -> Tensor1[tf32, Batch]:

    rho_ab_partial, rho_bc, rho_ac, prop_a = untransform_params(
        params_unknown_transformed,
    )

    log_likelihoods = log_likelihood(stats_tensor,
                                     rho_ab_partial, rho_bc, rho_ac, prop_a, n)
    if penalise_boundaries:
        log_likelihoods = penalise_log_likelihood(log_likelihoods,
                                                  params_unknown_transformed)

    return log_likelihoods


def log_likelihood_from_transformed_params_fixed_ab(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        n: Tensor1[tf32, Batch],
        rho_ab_partial_transformed: Tensor1[tf32, Batch],
        params_unknown_transformed: Tensor2[tf32, Batch, UnknownParams],
        penalise_boundaries: bool = True,
) -> Tensor1[tf32, Batch]:

    params_unknown_transformed = tf.concat([
        rho_ab_partial_transformed[:, None],
        params_unknown_transformed,
    ], axis=1)
    return log_likelihood_from_transformed_params(
        stats_tensor,
        n,
        params_unknown_transformed,
        penalise_boundaries,
    )


# Maybe a tf.function: see directly after function
def adam(
        likelihood_function,
        params_unknown_transformed_init: Tensor2[tf32, Batch, UnknownParams],
        m: Tensor2[tf32, Batch, UnknownParams],
        v: Tensor2[tf32, Batch, UnknownParams],
        t0: Tensor0[ti64],
        steps: Tensor0[ti64],
        lr_init: Tensor1[tf32, Batch],
        lr_decay: Union[Tensor0[tf32], Tensor1[tf32, Batch]],
        **hyperparameters: Tensor0,
) -> Tuple[Tensor2[tf32, Batch, UnknownParams],
           Tensor2[tf32, Batch, UnknownParams],
           Tensor2[tf32, Batch, UnknownParams],
           Tensor1[tf32, Batch]]:

    beta1 = hyperparameters['beta1']
    beta2 = hyperparameters['beta2']

    lr = lr_init
    params_unknown_transformed = params_unknown_transformed_init

    if UNROLL_WHEN_COMPILING:
        ts = range(int(steps))  # will create Python loop that gets unrolled.
    else:
        ts = tf.range(steps)  # will create a tf loop that doesn't get unrolled

    for t in ts:
        with tf.GradientTape() as tape:
            tape.watch(params_unknown_transformed)
            log_likelihood_per_item = likelihood_function(
                params_unknown_transformed
            )
            log_likelihood = tf.reduce_sum(log_likelihood_per_item)
        gradient = tape.gradient(log_likelihood, params_unknown_transformed)

        one = tf.constant(1.0, tf.float32)
        m = beta1 * m + (one - beta1) * gradient
        v = beta2 * v + (one - beta2) * tf.square(gradient)
        m_hat = m / (1. - beta1**tf.cast(t0 + t + 1, tf.float32))
        v_hat = v / (1. - beta2**tf.cast(t0 + t + 1, tf.float32))

        params_unknown_transformed += (lr[:, None] * m_hat
                                       / (tf.sqrt(v_hat) + 1e-8))
        lr *= lr_decay
    return params_unknown_transformed, m, v, log_likelihood_per_item


if not UNROLL_WHEN_COMPILING:
    adam = tf.function(adam, jit_compile=True)


def transform_correlations(
        correlations: Union[Tensor1, Tensor2],
) -> Union[Tensor1, Tensor2]:

    corrs_clipped = tf.clip_by_value(correlations, -MAX_ABS_RHO, MAX_ABS_RHO)
    corrs_tranformed = tf.math.atanh(corrs_clipped)
    return corrs_tranformed


def untransform_correlations(
    corrs_transformed: Union[Tensor1, Tensor2],
) -> Union[Tensor1, Tensor2]:

    correlations = tf.math.tanh(corrs_transformed)
    return tf.clip_by_value(correlations, -MAX_ABS_RHO, +MAX_ABS_RHO)


def transform_params(
    *params_list: Tensor1[tf32, Batch],
) -> Tensor2[tf32, Batch, UnknownParams]:

    prop_a = params_list[-1]
    prop_a_stretched = prop_a * 2. - 1.
    true_correlations_list = list(params_list[:-1])
    correlations = tf.stack(true_correlations_list + [prop_a_stretched],
                            axis=1)
    return transform_correlations(correlations)


def untransform_params(
        params_transformed: Tensor2[tf32, Batch, UnknownParams],
) -> Tuple[Tensor1[tf32, Batch], ...]:

    params_stretched = untransform_correlations(params_transformed)
    params_stretched = tf.unstack(params_stretched, axis=1)
    true_correlations = params_stretched[:-1]
    prop_a_stretched = params_stretched[-1]
    prop_a = (prop_a_stretched + 1.) / 2.

    return tuple(true_correlations + [prop_a])


def adam_and_polish(
        likelihood_function,
        params: Tensor2[tf32, Batch, UnknownParams],
        n: Tensor1[tf32, Batch],  # Just for setting learning rate.
        **hyperparameters: Tensor0,
) -> Tensor2[tf32, Batch, UnknownParams]:

    lr_init = hyperparameters['learning_rate_multiplier'] / n
    lr_decay = hyperparameters['learning_rate_polish_decay']
    iterations = hyperparameters['adam_iterations']
    iterations_polish = hyperparameters['adam_polish_iterations']

    m = tf.zeros_like(params)
    v = tf.zeros_like(params)

    params, m, v, _ = adam(
        likelihood_function,
        params, m, v, tf.constant(0, dtype=tf.int64),
        steps=iterations,
        lr_init=lr_init,
        lr_decay=tf.constant(1.0),
        **hyperparameters,
    )
    params, m, v, log_likelihood = adam(
        likelihood_function,
        params, m, v, iterations,
        steps=iterations_polish,
        lr_init=lr_init,
        lr_decay=lr_decay,
        **hyperparameters,
    )

    return params


def likelihood_ratios_via_gradient_ascent(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        rho_ab_partial_null: Tensor1[tf32, Batch],
        rho_ab_partial_power: Tensor1[tf32, Batch],
        rho_bc: Tensor1[tf32, Batch],
        rho_ac: Tensor1[tf32, Batch],
        prop_a: Tensor1[tf32, Batch],
        n: Tensor1[tf32, Batch],
        **hyperparameters,
) -> Tensor2[tf32, Batch, Ys]:

    params_unknown_transformed_init = transform_params(
        rho_ab_partial_null,
        rho_bc,
        rho_ac,
        prop_a,
    )
    rho_ab_partial_null_trans = transform_correlations(rho_ab_partial_null)
    rho_ab_partial_power_trans = transform_correlations(rho_ab_partial_power)

    log_likelihood_fn_alternative = functools.partial(
        log_likelihood_from_transformed_params,
        stats_tensor, n,
    )
    log_likelihood_fn_null = functools.partial(
        log_likelihood_from_transformed_params_fixed_ab,
        stats_tensor, n, rho_ab_partial_null_trans,
    )
    log_likelihood_fn_power_null = functools.partial(
        log_likelihood_from_transformed_params_fixed_ab,
        stats_tensor, n, rho_ab_partial_power_trans,
    )

    params_alt = adam_and_polish(
        log_likelihood_fn_alternative,
        params_unknown_transformed_init,
        n,
        **hyperparameters,
    )
    params_null = adam_and_polish(
        log_likelihood_fn_null,
        params_alt[:, 1:],
        n,
        **hyperparameters,
    )
    params_power_null = adam_and_polish(
        log_likelihood_fn_power_null,
        params_alt[:, 1:],
        n,
        **hyperparameters,
    )

    # Now we need to recover the **unpenalised** log likelihoods
    log_likelihoods_alt = log_likelihood_fn_alternative(
        params_alt,
        penalise_boundaries=False,
    )
    log_likelihoods_null = log_likelihood_fn_null(
        params_null,
        penalise_boundaries=False,
    )
    log_likelihoods_power_null = log_likelihood_fn_power_null(
        params_power_null,
        penalise_boundaries=False,
    )

    diffs = tf.stack([
        log_likelihoods_alt - log_likelihoods_null,
        log_likelihoods_alt - log_likelihoods_power_null,
    ], axis=1)

    return diffs


# TODO: Check that this is now reproducible even with JIT
@tf.function(jit_compile=True)
def likelihoods_for_batch(
    rho_ab_partial: Tensor1[tf32, Batch],
    rho_bc: Tensor1[tf32, Batch],
    rho_ac: Tensor1[tf32, Batch],
    prop_a: Tensor1[tf32, Batch],
    n: Tensor1[tf32, Batch],
    rho_ab_partial_power: Tensor1[tf32, Batch],
    params_num: Tensor0[ti32],
    first_row_num: Tensor0[ti32],
    num_rows: int,
    hyperparameter_overrides: Optional[Dict[str, Tensor0]] = None,
) -> Tensor2[tf32, Batch, Ys]:

    if hyperparameter_overrides is None:
        hyperparameters = HYPERPARAMETERS_DEFAULT
    else:
        hyperparameters = HYPERPARAMETERS_DEFAULT | hyperparameter_overrides

    stats = biparcorr.sampling_distribution_fn_raw(
        rho_ab_partial=rho_ab_partial,
        rho_bc=rho_bc,
        rho_ac=rho_ac,
        prop_a=prop_a,
        n=n,
        params_num=params_num,
        first_row_num=first_row_num,
        num_rows=num_rows,
    )

    log_likelihoods = log_likelihoods_via_gradient_ascent(stats,
                                                         rho_ab_partial,
                                                         rho_ab_partial_power,
                                                         rho_bc,
                                                         rho_ac,
                                                         prop_a,
                                                         n,
                                                         **hyperparameters)

    return log_likelihoods


def ps_for_batch(
        rho_ab_partial: Tensor1[tf32, Batch],
        rho_bc: Tensor1[tf32, Batch],
        rho_ac: Tensor1[tf32, Batch],
        prop_a: Tensor1[tf32, Batch],
        n: Tensor1[tf32, Batch],
        rho_ab_partial_power: Tensor1[tf32, Batch],
        batch_size: int,
        hyperparameter_overrides: Optional[Dict[str, Tensor0]] = None,
) -> Tensor2[tf32, Batch, Ys]:

    likelihoods = likelihoods_for_batch(
        rho_ab_partial,
        rho_bc,
        rho_ac,
        prop_a,
        n,
        rho_ab_partial_power,
        batch_size,
        hyperparameter_overrides,
    )
    diffs = tf.stack([
        likelihoods[:, 0] - likelihoods[:, 1],
        likelihoods[:, 0] - likelihoods[:, 2],
    ], axis=1)
    ps_lr = tf.math.igammac(0.5, 0.5 * 2.0 * diffs)
    return ps_lr
