import biparcorr_analyse_funcs as biparcorr

import tensorflow as tf
import tensorflow_probability as tfp
tfd, tfb = tfp.distributions, tfp.bijectors

import functools

# typing
from typing import Optional, Union, Tuple
from neuralcis.common import Batch, Samples, Stats, UnknownParams, Ys
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3
from tensor_annotations.tensorflow import float32 as tf32, int64 as ti64


def log_likelihood(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        rho_ab_partial: Tensor1[tf32, Batch],
        rho_bc: Tensor1[tf32, Batch],
        rho_ac: Tensor1[tf32, Batch],
        prop_a: Tensor1[tf32, Batch],
        n: Tensor0[ti64],
) -> Tensor1[tf32, Batch]:

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
    mvn_bc = tfd.MultivariateNormalTriL(
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


def log_likelihood_from_transformed_params(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        n: Tensor0[ti64],
        params_unknown_transformed: Tensor2[tf32, Batch, UnknownParams],
) -> Tensor1[tf32, Batch]:

    rho_ab_partial, rho_bc, rho_ac, prop_a = untransform_params(
        params_unknown_transformed,
        includes_rho_ab_partial=True,
    )
    return log_likelihood(stats_tensor,
                          rho_ab_partial, rho_bc, rho_ac, prop_a,
                          n)


def log_likelihood_from_transformed_params_fixed_ab(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        n: Tensor0[ti64],
        rho_ab_partial: Tensor1[tf32, Batch],
        params_unknown_transformed: Tensor2[tf32, Batch, UnknownParams],
) -> Tensor1[tf32, Batch]:

    rho_bc, rho_ac, prop_a = untransform_params(params_unknown_transformed,
                                                includes_rho_ab_partial=False)

    return log_likelihood(stats_tensor,
                          rho_ab_partial, rho_bc, rho_ac, prop_a,
                          n)


@tf.function(jit_compile=True)
def adam(
        likelihood_function,
        params_unknown_transformed_init: Tensor2[tf32, Batch, UnknownParams],
        m: Tensor2[tf32, Batch, UnknownParams],
        v: Tensor2[tf32, Batch, UnknownParams],
        t0: Tensor0[ti64],
        steps: int,
        lr_init: Tensor0[tf32],
        lr_decay: Tensor0[tf32],
) -> Tuple[Tensor2[tf32, Batch, UnknownParams],
           Tensor2[tf32, Batch, UnknownParams],
           Tensor2[tf32, Batch, UnknownParams],
           Tensor1[tf32, Batch]]:

    beta1 = 0.8
    beta2 = 0.99

    lr = lr_init
    params_unknown_transformed = params_unknown_transformed_init
    for t in range(steps):  # Deliberately a Python loop; gets unrolled.
        with tf.GradientTape() as tape:
            tape.watch(params_unknown_transformed)
            log_likelihood_per_item = likelihood_function(
                params_unknown_transformed
            )
            log_likelihood = tf.reduce_sum(log_likelihood_per_item)
        gradient = tape.gradient(log_likelihood, params_unknown_transformed)

        m = beta1 * m + (1-beta1) * gradient
        v = beta2 * v + (1-beta2) * tf.square(gradient)
        m_hat = m / (1. - beta1**tf.cast(t0 + t + 1, tf.float32))
        v_hat = v / (1. - beta2**tf.cast(t0 + t + 1, tf.float32))

        params_unknown_transformed += lr * m_hat / (tf.sqrt(v_hat) + 1e-8)
        lr *= lr_decay
    return params_unknown_transformed, m, v, log_likelihood_per_item


def transform_params(
    rho_ab_partial: Optional[Tensor1[tf32, Batch]],
    rho_bc: Tensor1[tf32, Batch],
    rho_ac: Tensor1[tf32, Batch],
    prop_a: Tensor1[tf32, Batch],
) -> Tensor2[tf32, Batch, UnknownParams]:

    prop_a_stretched = prop_a * 2. - 1.  # -> [-1, 1] like a correlation
    if rho_ab_partial is None:
        params = tf.stack([
            rho_bc,
            rho_ac,
            prop_a_stretched,
        ], axis=1)
    else:
        params = tf.stack([
            rho_ab_partial,
            rho_bc,
            rho_ac,
            prop_a_stretched,
        ], axis=1)
    params_clipped = tf.clip_by_value(params, -0.98, 0.98)
    params_transformed = tf.math.atanh(params_clipped / 0.99)
    return params_transformed


def untransform_params(
        params_transformed: Tensor2[tf32, Batch, UnknownParams],
        includes_rho_ab_partial: bool = True,
) -> Union[Tuple[Tensor1[tf32, Batch],
                 Tensor1[tf32, Batch],
                 Tensor1[tf32, Batch],
                 Tensor1[tf32, Batch]],
           Tuple[Tensor1[tf32, Batch],
                 Tensor1[tf32, Batch],
                 Tensor1[tf32, Batch]]]:

    params = tf.math.tanh(params_transformed) * 0.99
    if includes_rho_ab_partial:
        (rho_ab_partial,
         rho_bc, rho_ac, prop_a_stretched) = tf.unstack(params, axis=1)
    else:
        rho_bc, rho_ac, prop_a_stretched = tf.unstack(params, axis=1)

    prop_a = (prop_a_stretched + 1.) / 2.

    if includes_rho_ab_partial:
        return rho_ab_partial, rho_bc, rho_ac, prop_a
    else:
        return rho_bc, rho_ac, prop_a


def multistep_adam(
        likelihood_function,
        params: Tensor2[tf32, Batch, UnknownParams],
        n_float: Tensor0[tf32],
) -> Tensor2[tf32, Batch, Ys]:

    lr_init = 1./n_float
    m = tf.zeros_like(params)
    v = tf.zeros_like(params)
    tf_int = lambda t0: tf.constant(t0, dtype=tf.int64)

    params40, m40, v40, log_likelihood40 = adam(
        likelihood_function,
        params, m, v, tf_int(0),
        steps=40,
        lr_init=lr_init,
        lr_decay=tf.constant(1.0),
    )
    params80, m80, v80, log_likelihood80 = adam(
        likelihood_function,
        params40, m40, v40, tf_int(40),
        steps=40,
        lr_init=lr_init,
        lr_decay=tf.constant(1.0),
    )
    params120, m120, v120, log_likelihood120 = adam(
        likelihood_function,
        params80, m80, v80, tf_int(80),
        steps=40,
        lr_init=lr_init,
        lr_decay=tf.constant(1.0),
    )
    _, _, _, log_likelihood120_40 = adam(
        likelihood_function,
        params120, m120, v120, tf_int(120),
        steps=40,
        lr_init=lr_init,
        lr_decay=tf.constant(0.9),
    )

    return tf.stack([log_likelihood40,
                     log_likelihood80,
                     log_likelihood120,
                     log_likelihood120_40], axis=1)


def likelihood_ratios_via_gradient_ascent(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        rho_ab_partial_null: Tensor0[tf32],
        rho_ab_partial_power: Tensor0[tf32],
        n: Tensor0[ti64],
) -> Tensor2[tf32, Batch, Ys]:

    n_float = tf.cast(n, tf.float32)

    log_likelihood_fn_alternative = functools.partial(
        log_likelihood_from_transformed_params,
        stats_tensor, n,
    )
    log_likelihood_fn_null = functools.partial(
        log_likelihood_from_transformed_params_fixed_ab,
        stats_tensor, n, rho_ab_partial_null,
    )
    log_likelihood_fn_power_null = functools.partial(
        log_likelihood_from_transformed_params_fixed_ab,
        stats_tensor, n, rho_ab_partial_power,
    )

    # use sample correlations as basis for initial guess
    correlations_hat = biparcorr.estimate_correlations_safe(stats_tensor,
                                                            n_float)
    prop_a_hat = tf.reduce_sum(stats_tensor[:, :, 0], axis=1) / n_float

    rho_ab_hat, rho_bc_hat, rho_ac_hat = tf.unstack(correlations_hat, axis=1)
    rho_ab_partial_hat = ((rho_ab_hat - rho_bc_hat*rho_ac_hat) /
                          tf.sqrt((1. - tf.square(rho_bc_hat))
                                  * (1. - tf.square(rho_ac_hat))))

    params_unknown_transformed_init = transform_params(rho_ab_partial_hat,
                                                       rho_bc_hat,
                                                       rho_ac_hat,
                                                       prop_a_hat)

    log_likelihoods_null = multistep_adam(
        log_likelihood_fn_null,
        params_unknown_transformed_init[:, 1:],
        n_float
    )
    log_likelihoods_alt = multistep_adam(
        log_likelihood_fn_alternative,
        params_unknown_transformed_init,
        n_float,
    )
    log_likelihoods_power_null = multistep_adam(
        log_likelihood_fn_power_null,
        params_unknown_transformed_init[:, 1:],
        n_float,
    )

    diffs = tf.concat([
        log_likelihoods_alt - log_likelihoods_null,
        log_likelihoods_alt[:, -1:] - log_likelihoods_power_null[:, -1:],
    ], axis=1)

    return diffs


def negative_log_likelihood_from_transformed_params(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        n: Tensor0[ti64],
        params_unknown_transformed: Tensor2[tf32, Batch, UnknownParams],
) -> Tensor1[tf32, Batch]:

    return -log_likelihood_from_transformed_params(
        stats_tensor,
        n,
        params_unknown_transformed,
    )


def negative_log_likelihood_from_transformed_params_fixed_ab(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        n: Tensor0[ti64],
        rho_ab_partial: Tensor1[tf32, Batch],
        params_unknown_transformed: Tensor2[tf32, Batch, UnknownParams],
) -> Tensor1[tf32, Batch]:

    return -log_likelihood_from_transformed_params_fixed_ab(
        stats_tensor,
        n,
        rho_ab_partial,
        params_unknown_transformed
    )


def likelihood_ratio_via_line_search(
        rho_ab_partial_null: Tensor0[tf32],
        rho_ab_partial_power: Tensor0[tf32],
        n: Tensor0[ti64],
        stats_tensor: Tensor2[tf32, Samples, Stats],
) -> Tensor1[tf32, Ys]:

    n_float = tf.cast(n, tf.float32)

    log_likelihood_fn_alternative = functools.partial(
        negative_log_likelihood_from_transformed_params,
        stats_tensor[None], n,
    )
    log_likelihood_fn_null = functools.partial(
        negative_log_likelihood_from_transformed_params_fixed_ab,
        stats_tensor[None], n, rho_ab_partial_null,
    )
    log_likelihood_fn_power_null = functools.partial(
        negative_log_likelihood_from_transformed_params_fixed_ab,
        stats_tensor[None], n, rho_ab_partial_power,
    )

    # use sample correlations as basis for initial guess
    correlations_hat = biparcorr.estimate_correlations_safe(stats_tensor[None],
                                                            n_float)
    prop_a_hat = tf.reduce_sum(stats_tensor[:, 0], keepdims=True) / n_float

    rho_ab_hat, rho_bc_hat, rho_ac_hat = tf.unstack(correlations_hat, axis=1)
    rho_ab_partial_hat = ((rho_ab_hat - rho_bc_hat*rho_ac_hat) /
                          tf.sqrt((1. - tf.square(rho_bc_hat))
                                  * (1. - tf.square(rho_ac_hat))))

    params_unknown_transformed_init = transform_params(rho_ab_partial_hat,
                                                       rho_bc_hat,
                                                       rho_ac_hat,
                                                       prop_a_hat)

    res_alt = tfp.optimizer.lbfgs_minimize(
        lambda stats: tfp.math.value_and_gradient(
            log_likelihood_fn_alternative,
            stats,
        ),
        params_unknown_transformed_init,
    )
    lr_alt = -res_alt.objective_value

    res_null = tfp.optimizer.lbfgs_minimize(
        lambda stats: tfp.math.value_and_gradient(
            log_likelihood_fn_null,
            stats,
        ),
        params_unknown_transformed_init[:, 1:],
    )
    lr_null = -res_null.objective_value

    res_power_null = tfp.optimizer.lbfgs_minimize(
        lambda stats: tfp.math.value_and_gradient(
            log_likelihood_fn_power_null,
            stats,
        ),
        params_unknown_transformed_init[:, 1:],
    )
    lr_power_null = -res_power_null.objective_value

    return tf.stack([lr_alt[0] - lr_null[0], lr_alt[0] - lr_power_null[0]])


@tf.function(jit_compile=True)
def ps_for_batch(
    rho_ab_partial: Tensor0[tf32],
    rho_bc: Tensor0[tf32],
    rho_ac: Tensor0[tf32],
    prop_a: Tensor0[tf32],
    n: Tensor0[ti64],
    rho_ab_partial_power: Tensor0[tf32],
    batch_size: int,
    generator: tf.random.Generator,
    do_l_bfgs: bool,
) -> Tensor2[tf32, Batch, Ys]:

    stats = biparcorr.sampling_distribution_fn_raw(
        rho_ab_partial=rho_ab_partial,
        rho_bc=rho_bc,
        rho_ac=rho_ac,
        prop_a=prop_a,
        n=n,
        batch_size=batch_size,
        generator=generator,
    )

    diffs = likelihood_ratios_via_gradient_ascent(stats,
                                                  rho_ab_partial,
                                                  rho_ab_partial_power,
                                                  n)

    if do_l_bfgs:
        diff_bfgs = tf.map_fn(
            functools.partial(likelihood_ratio_via_line_search,
                              rho_ab_partial, rho_ab_partial_power, n),
            stats,
            parallel_iterations=batch_size,
            fn_output_signature=tf.TensorSpec([2], tf.float32),
        )
        diffs = tf.concat([diffs, diff_bfgs], axis=1)

    ps_lr = tf.math.igammac(0.5, 0.5 * 2.0*diffs)

    return ps_lr
