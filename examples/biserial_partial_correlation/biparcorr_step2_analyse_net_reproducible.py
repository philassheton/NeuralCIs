import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from neuralcis import NeuralCIs

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd, tfb = tfp.distributions, tfp.bijectors

from tqdm import tqdm
from datetime import datetime
import time
import functools

# typing
from typing import Optional, Union, Tuple
from neuralcis.common import Batch, Samples, Stats, UnknownParams, One, Ys
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3
from tensor_annotations.tensorflow import float32 as tf32, int64 as ti64


# No need for N_MIN as they are sampled from the network
N_MAX = 100
DO_L_BFGS = True
NUM_PARAM_SAMPLES = 1_000
NUM_SIMULATIONS_PER_PARAM = 100_000 # 10_000_000
BATCH_SIZE = 1_000


def sampling_distribution_fn_raw(
        rho_ab_partial: Tensor0[tf32],
        rho_bc: Tensor0[tf32],
        rho_ac: Tensor0[tf32],
        prop_a: Tensor0[tf32],
        n: Tensor0[ti64],
        batch_size: int,
) -> Tensor3[tf32, Batch, Samples, Stats]:

    rho_ab = (rho_bc * rho_ac
              + rho_ab_partial * tf.sqrt((1. - tf.square(rho_bc)) *
                                         (1. - tf.square(rho_ac))))

    z_threshold_a = -tfp.distributions.Normal(0., 1.).quantile(prop_a)

    one = tf.ones_like(rho_ab)
    correlation_matrix = tf.stack([
        tf.stack([one, rho_ab, rho_ac], axis=0),
        tf.stack([rho_ab, one, rho_bc], axis=0),
        tf.stack([rho_ac, rho_bc, one], axis=0),
    ], axis=1)

    cholesky = tf.linalg.cholesky(correlation_matrix)

    z = tf.random.normal((batch_size, 3, N_MAX)) * n_mask(n)[None, None, :]
    z_correlated = tf.linalg.matmul(cholesky, z)

    a, b, c = tf.split(z_correlated, 3, axis=1)
    a = tf.cast(a > z_threshold_a, tf.float32) * n_mask(n)[None, None, :]
    samples = tf.stack([a[:, 0, :], b[:, 0, :], c[:, 0, :]], axis=2)

    return samples


# n_mask ensures we only have n z values (the rest will be zeroed)
#  -- this means we work with constant memory size.
def n_mask(n: Tensor0[ti64]) -> Tensor1[tf32, Samples]:
    return tf.cast(tf.range(N_MAX, dtype=tf.int64) < n, tf.float32)


def n_mask_safe(n: Tensor0) -> Tensor1[tf32, Samples]:
    return n_mask(tf.cast(n, tf.int64))


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

    log_prob_abc_masked = n_mask(n) * (log_prob_a + log_prob_bc)

    log_likelihood_final = tf.reduce_sum(log_prob_abc_masked, axis=1)
    return log_likelihood_final


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


def estimate_correlations_safe(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        n_float: Tensor0[tf32],
) -> Tensor2[tf32, Batch, 3]:

    # Returns zero correlation whenever all a values are the same.

    stats_tensor = stats_tensor * n_mask_safe(n_float)[None, :, None]          # type: ignore
    stats_mean = tf.reduce_sum(stats_tensor, axis=1, keepdims=True) / n_float
    X = (stats_tensor - stats_mean) * n_mask_safe(n_float)[None, :, None]

    X_sum_pairwise = tf.reduce_sum(X * tf.gather(X, [1, 2, 0], axis=2), axis=1)
    X_sum_sq = tf.reduce_sum(tf.square(X), axis=1)
    X_sum_sq_pairwise = X_sum_sq * tf.gather(X_sum_sq, [1, 2, 0], axis=1)

    correlations_hat = X_sum_pairwise / tf.sqrt(X_sum_sq_pairwise + 1e-10)

    return correlations_hat


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
    correlations_hat = estimate_correlations_safe(stats_tensor, n_float)
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

    return -log_likelihood_from_transformed_params(stats_tensor,
                                                   n,
                                                   params_unknown_transformed)


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
    correlations_hat = estimate_correlations_safe(stats_tensor[None], n_float)
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


def ps_neural_tf(
        cis: NeuralCIs,
        stats: Tensor3[tf32, Batch, Samples, Stats],
        batch_size: int,
        **params: Tensor0[tf32],
):

    params_human = {name: tf.repeat(param, (batch_size,))
                    for name, param in params.items()}

    correlations = estimate_correlations_safe(stats, params['n'])
    rho_ab_hat, rho_bc_hat, rho_ac_hat = tf.unstack(correlations, axis=1)
    prop_a_hat = tf.reduce_sum(stats[:, :, 0], axis=1) / params['n']

    params_net = cis._params_human_to_net(**params_human)
    stats_net = cis._stats_human_to_net(
        rho_ab_hat=rho_ab_hat,
        rho_bc_hat=rho_bc_hat,
        rho_ac_hat=rho_ac_hat,
        prop_a_hat=prop_a_hat,
    )

    ps = cis.pnet.p(stats_net, params_net)

    return ps


@tf.function(jit_compile=True)
def ps_for_batch(
    rho_ab_partial: Tensor0[tf32],
    rho_bc: Tensor0[tf32],
    rho_ac: Tensor0[tf32],
    prop_a: Tensor0[tf32],
    n: Tensor0[ti64],
    rho_ab_partial_power: Tensor0[tf32],
    cis: NeuralCIs,
    batch_size: int,
) -> Tensor2[tf32, Batch, Ys]:

    stats = sampling_distribution_fn_raw(
        rho_ab_partial=rho_ab_partial,
        rho_bc=rho_bc,
        rho_ac=rho_ac,
        prop_a=prop_a,
        n=n,
        batch_size=batch_size,
    )

    diffs = likelihood_ratios_via_gradient_ascent(stats,
                                                  rho_ab_partial,
                                                  rho_ab_partial_power,
                                                  n)

    if DO_L_BFGS:
        diff_bfgs = tf.map_fn(
            functools.partial(likelihood_ratio_via_line_search,
                              rho_ab_partial, rho_ab_partial_power, n),
            stats,
            parallel_iterations=batch_size,
            fn_output_signature=tf.TensorSpec([2], tf.float32),
        )
        diffs = tf.concat([diffs, diff_bfgs], axis=1)

    ps_lr = tf.math.igammac(0.5, 0.5 * 2.0*diffs)

    ps_neural_null = ps_neural_tf(cis, stats, batch_size,
                                  rho_ab_partial=rho_ab_partial,
                                  rho_bc=rho_bc,
                                  rho_ac=rho_ac,
                                  prop_a=prop_a,
                                  n=tf.cast(n, tf.float32))
    ps_neural_power = ps_neural_tf(cis, stats, batch_size,
                                   rho_ab_partial=rho_ab_partial_power,
                                   rho_bc=rho_bc,
                                   rho_ac=rho_ac,
                                   prop_a=prop_a,
                                   n=tf.cast(n, tf.float32))
    ps_neural = tf.stack([ps_neural_null, ps_neural_power], axis=1)

    ps = tf.concat([ps_neural, ps_lr], axis=1)

    return ps


def make_cdf_summary(
        ps: Tensor2[tf32, Samples, Stats],
        summary_length: int,
) -> Tensor3[tf32, One, Stats, Samples]:

    ps_sorted = tf.sort(tf.transpose(ps), axis=1)
    end_of_bucket_index = tf.linspace(0., len(ps), summary_length + 1)[1:] - 1
    end_of_bucket_index = tf.cast(end_of_bucket_index, tf.int32)
    return tf.gather(ps_sorted, end_of_bucket_index, axis=1)[None, :, :]


def rho_for_power(
        n: Tensor0[ti64],
        prop_a: Tensor0[tf32],
        rho_ab_partial_null: Tensor0[tf32],
        power: float = 0.80,
        alpha: float = 0.05,
        k: int = 1,
        alt_sign: int = 1,
):

    normal = tfp.distributions.Normal(0., 1.)

    power = tf.constant(power, dtype=tf.float32)
    alpha = tf.constant(alpha, dtype=tf.float32)
    k = tf.constant(k, dtype=tf.float32)
    alt_sign = tf.constant(float(tf.sign(alt_sign)), tf.float32)
    n = tf.cast(n, tf.float32)

    zcrit = normal.quantile(1.0 - alpha / 2.0)
    zbeta = normal.quantile(power)
    delta_z = (zcrit + zbeta) / tf.sqrt(n - k - 3.0)

    # ---- helper: latent ρ  <-->  observed point-biserial r ----
    z_pi = normal.quantile(1.0 - prop_a)
    phi = normal.prob(z_pi)
    scale = phi / tf.sqrt(prop_a * (1.0 - prop_a))  # r = ρ * scale
    invscale = 1.0 / scale  # ρ = r * invscale

    # Null on observed scale
    r0 = rho_ab_partial_null * scale
    # Fisher z of null
    z0 = tf.atanh(tf.clip_by_value(r0, -0.999999, 0.999999))

    # Alternative Fisher z
    z1 = z0 + alt_sign * delta_z
    # Back to observed r, then to latent ρ
    r1 = tf.tanh(z1)
    rho1 = r1 * invscale

    # Avoid overflow outside (-1,1) due to numeric noise
    rho1 = tf.clip_by_value(rho1, -0.999999, 0.999999)
    return rho1


def profile_pvalues_for_one_params(
        params_human: dict[str, Tensor0[tf32]],
        rho_ab_partial_power: Tensor0[tf32],
        num_simulations: int,
        cdf_summary_length: int = 1000,
        batch_size: int = 500,
):

    n = tf.cast(params_human['n'], tf.int64)

    ps = []
    num_batches = num_simulations // batch_size
    start = time.perf_counter()
    for batch_num in tqdm(range(num_batches)):
        ps_batch = ps_for_batch(
            params_human['rho_ab_partial'],
            params_human['rho_bc'],
            params_human['rho_ac'],
            params_human['prop_a'],
            n,
            rho_ab_partial_power,
            cis,
            batch_size,
        )

        # CPU!!!


        ps.append(ps_batch)
    ps = tf.concat(ps, axis=0)

    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} seconds")
    print(f"Processed parameters: {params_human}")
    return make_cdf_summary(ps, cdf_summary_length)


def profile_and_save_pvalue_summaries_for_params(
        save_directory: str,
        params_human: dict[str, Tensor1[tf32, Samples]],
        num_param_samples: int,
        num_simulations_per_param: int,
        cdf_summary_length: int = 1000,
        batch_size: int = 500,
) -> None:

    for params_num in range(num_param_samples):
        this_params = {name: param[params_num]
                       for name, param in params_human.items()}

        target_power = np.random.choice([0.50, 0.80])
        sign_power = np.random.choice([-1, +1])
        n = tf.cast(this_params['n'], tf.int64)
        rho_ab_partial_power = rho_for_power(n,
                                             this_params['prop_a'],
                                             this_params['rho_ab_partial'],
                                             power=target_power,
                                             alpha=0.05,
                                             alt_sign=sign_power)

        summary_tensor = profile_pvalues_for_one_params(
            params_human=this_params,
            rho_ab_partial_power=rho_ab_partial_power,
            num_simulations=num_simulations_per_param,
            cdf_summary_length=cdf_summary_length,
            batch_size=batch_size,
        )
        power_text = f'{int(target_power*100):d}{sign_power:+d}'.rstrip('1')
        name_start = f'neur POW lr40 80 120 120_40 POW'
        if DO_L_BFGS:
            name_start += ' bfgs POW'
        name_start += f'{power_text}'
        summary_name = (f'{save_directory}/{name_start} '
                        f'{datetime.now().strftime("%Y%m%d %H%M%S")}'
                        f' r_ab_p {this_params["rho_ab_partial"]:.4f}'
                        f' r_bc {this_params["rho_bc"]:.4f}'
                        f' r_ac {this_params["rho_ac"]:.4f}'
                        f' p_a {this_params["prop_a"]:.4f}'
                        f' n {int(this_params["n"]):d}'
                        f' runs {num_simulations_per_param}')
        np.save(summary_name, summary_tensor.numpy())


save_directory = 'param_runs'
os.makedirs(save_directory, exist_ok=True)
cis = NeuralCIs.load('saved_model',
                     'testing')
params = cis.sample_params(NUM_PARAM_SAMPLES)

profile_and_save_pvalue_summaries_for_params(
    save_directory=save_directory,
    params_human=params,
    num_param_samples=NUM_PARAM_SAMPLES,
    num_simulations_per_param=NUM_SIMULATIONS_PER_PARAM,
    cdf_summary_length=1000,
    batch_size=BATCH_SIZE,
)
