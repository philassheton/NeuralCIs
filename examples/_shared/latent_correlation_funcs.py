import tensorflow as tf
import tensorflow_probability as tfp

import functools

# typing
from typing import Optional, Tuple, Callable, Union
from collections.abc import Sequence
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3
from tensor_annotations.tensorflow import float32 as tf32, int32 as ti32
from .common import Batch, Samples
from .common import Stats, PairwiseCorrelations, UnknownParams, Ys


N_MAX = 100
MAX_ABS_RHO = 0.99
SEED_FOR_BATCH_INCONSISTENT = 12345


def generate_random_normals(
        num_rows: int,
        row_shape: Tuple[int, int],
        simulation_block_num: Tensor0[ti32],
        batch_consistent_randoms: bool = False,
        first_row_num: Optional[Tensor0[ti32]] = None,
) -> Tensor3:

    if batch_consistent_randoms:
        return generate_random_normals_batch_consistent(num_rows, row_shape,
                                                        simulation_block_num,
                                                        first_row_num)
    else:
        return generate_random_normals_batch_inconsistent(num_rows, row_shape,
                                                          simulation_block_num)


def generate_random_normals_batch_consistent(
        num_rows: int,
        row_shape: Tuple[int, int],
        simulation_block_num: Tensor0[ti32],
        first_row_num: Optional[Tensor0[ti32]] = None,
) -> Tensor3:

    row_nums = tf.range(num_rows) + first_row_num
    return tf.map_fn(
        lambda row: tf.random.stateless_normal(
            row_shape,
            tf.stack([simulation_block_num, row])
        ),
        row_nums,
        dtype=tf.float32,
    )


def generate_random_normals_batch_inconsistent(
        num_rows: int,
        row_shape: Tuple[int, int],
        simulation_block_num: Tensor0[ti32],
) -> Tensor3:

    shape = [num_rows] + list(row_shape)
    return tf.random.stateless_normal(
        shape,
        tf.stack([simulation_block_num, SEED_FOR_BATCH_INCONSISTENT]),
    )


# n_mask ensures we only have n z values (the rest will be zeroed)
#  -- this means we work with constant memory size.
def n_mask(n: Tensor1[tf32, Batch]) -> Tensor2[tf32, Batch, Samples]:
    n = tf.cast(n, tf.int64)
    return tf.cast(tf.sequence_mask(n, N_MAX), tf.float32)


def sampling_distribution_fn_raw_generic(
        prop_a_b_c: Sequence[Optional[Tensor1[tf32, Batch]]],
        n: Tensor1[tf32, Batch],
        simulation_block_num: Tensor0[ti32],
        num_rows: int,

        rho_ab_partial: Optional[Tensor1[tf32, Batch]] = None,
        rho_ab: Optional[Tensor1[tf32, Batch]] = None,
        rho_bc: Optional[Tensor1[tf32, Batch]] = None,
        rho_ac: Optional[Tensor1[tf32, Batch]] = None,


        # Using this batch_consistent_randoms option will mean that a batched
        # process can generate reproducible stats that can be compared across
        # different runs.  If set to True, must also provide first_row_num
        # to locate the batch within the list of simulations for this
        # simulation block.  However, this is also much slower.
        batch_consistent_randoms: bool = False,
        first_row_num_within_simulation_block: Optional[Tensor0[ti32]] = None,

        is_powersim_run: bool = False,
        is_bootstrap_simulation: bool = False,
) -> Tensor3[tf32, Batch, Samples, Stats]:

    if batch_consistent_randoms:
        assert first_row_num_within_simulation_block is not None
    else:
        assert first_row_num_within_simulation_block is None

    if is_powersim_run:
        simulation_block_num += 1_000_000_000
    if is_bootstrap_simulation:
        simulation_block_num += 2_000_000_000

    if rho_ab_partial is not None:
        assert rho_ab is None
        rho_ab = (rho_bc * rho_ac
                  + rho_ab_partial * tf.sqrt((1. - tf.square(rho_bc)) *
                                             (1. - tf.square(rho_ac))))

    z_threshold_a_b_c = [
        -tfp.distributions.Normal(0., 1.).quantile(prop) if prop is not None
        else None
        for prop in prop_a_b_c
    ]

    one = tf.ones_like(rho_ab)

    if rho_ac is not None:
        assert rho_bc is not None
        num_variables = 3
        correlation_matrix = tf.stack([
            tf.stack([one, rho_ab, rho_ac], axis=-1),
            tf.stack([rho_ab, one, rho_bc], axis=-1),
            tf.stack([rho_ac, rho_bc, one], axis=-1),
        ], axis=-1)
    else:
        num_variables = 2
        correlation_matrix = tf.stack([
            tf.stack([one, rho_ab], axis=-1),
            tf.stack([rho_ab, one], axis=-1),
        ], axis=-1)

    cholesky = tf.linalg.cholesky(correlation_matrix)

    mask = n_mask(n)[:, None, :]
    z = generate_random_normals(
        num_rows,
        (num_variables, N_MAX),
        simulation_block_num,
        batch_consistent_randoms,
        first_row_num_within_simulation_block,
    ) * mask
    z_correlated = tf.linalg.matmul(cholesky, z)

    a_b_c = tf.unstack(z_correlated, axis=1)
    samples_thresholded = tf.stack([
        tf.cast(var > z_threshold[:, None], tf.float32) * mask[:, 0, :]
        if z_threshold is not None
        else var
        for var, z_threshold in zip(a_b_c, z_threshold_a_b_c)
    ], axis=2)

    return samples_thresholded


def estimate_correlations_safe(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        n: Tensor1[tf32, Batch],
) -> Tensor2[tf32, Batch, PairwiseCorrelations]:

    # Returns zero correlation whenever all a values are the same.

    stats_tensor = stats_tensor * n_mask(n)[:, :, None]                        # type: ignore
    stats_mean = tf.reduce_sum(stats_tensor,
                               axis=1,
                               keepdims=True) / n[:, None, None]
    X = (stats_tensor - stats_mean) * n_mask(n)[:, :, None]

    X_sum_pairwise = tf.reduce_sum(X * tf.gather(X, [1, 2, 0], axis=2), axis=1)
    X_sum_sq = tf.reduce_sum(tf.square(X), axis=1)
    X_sum_sq_pairwise = X_sum_sq * tf.gather(X_sum_sq, [1, 2, 0], axis=1)

    correlations_hat = X_sum_pairwise / tf.sqrt(X_sum_sq_pairwise + 1e-10)

    return correlations_hat


def estimate_prop_a(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        n: Tensor1[tf32, Batch],
) -> Tensor1[tf32, Batch]:

    a = stats_tensor[:, :, 0] * n_mask(n)
    num_a = tf.reduce_sum(a, axis=1)
    prop_a_hat = num_a / n
    return prop_a_hat


def penalise_log_likelihood(
        log_likelihoods: Tensor1[tf32, Batch],
        params_unknown_transformed: Tensor2[tf32, Batch, UnknownParams],
) -> Tensor1[tf32, Batch]:

    max_abs_rho_transformed = transform_correlations(tf.constant(MAX_ABS_RHO))
    max_abs = tf.reduce_max(tf.math.abs(params_unknown_transformed), axis=1)
    penalty_unscaled = tf.nn.relu(max_abs - max_abs_rho_transformed)

    return log_likelihoods - 10.*penalty_unscaled


def log_likelihood_from_transformed_params(
        log_likelihood_fn: Callable,
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        n: Tensor1[tf32, Batch],
        params_unknown_transformed: Tensor2[tf32, Batch, UnknownParams],
        penalise_boundaries: bool = True,
) -> Tensor1[tf32, Batch]:

    rho_ab_partial, rho_bc, rho_ac, prop_a = untransform_params(
        params_unknown_transformed,
        param_is_proportion=(False, False, False, True),
    )

    log_likelihoods = log_likelihood_fn(
        stats_tensor,
        rho_ab_partial, rho_bc, rho_ac, prop_a, n,
    )
    if penalise_boundaries:
        log_likelihoods = penalise_log_likelihood(log_likelihoods,
                                                  params_unknown_transformed)

    return log_likelihoods


def log_likelihood_from_transformed_params_fixed_ab(
        log_likelihood_fn: Callable,
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
        log_likelihood_fn,
        stats_tensor,
        n,
        params_unknown_transformed,
        penalise_boundaries,
    )


def transform_correlations(
        correlations: Union[Tensor0, Tensor1, Tensor2],
) -> Union[Tensor0, Tensor1, Tensor2]:

    corrs_clipped = tf.clip_by_value(correlations, -MAX_ABS_RHO, MAX_ABS_RHO)
    corrs_tranformed = tf.math.atanh(corrs_clipped)
    return corrs_tranformed


def untransform_correlations(
    corrs_transformed: Union[Tensor1, Tensor2],
) -> Union[Tensor1, Tensor2]:

    correlations = tf.math.tanh(corrs_transformed)
    return tf.clip_by_value(correlations, -MAX_ABS_RHO, +MAX_ABS_RHO)


def transform_params(
    rho_ab_possibly_partial: Tensor1[tf32, Batch],
    rho_bc: Optional[Tensor1[tf32, Batch]] = None,
    rho_ac: Optional[Tensor1[tf32, Batch]] = None,
    prop_a: Optional[Tensor1[tf32, Batch]] = None,
    prop_b: Optional[Tensor1[tf32, Batch]] = None,
    prop_c: Optional[Tensor1[tf32, Batch]] = None,
) -> Tensor2[tf32, Batch, UnknownParams]:

    to_transform = [rho_ab_possibly_partial]
    if rho_bc is not None:
        to_transform.append(rho_bc)
    if rho_ac is not None:
        to_transform.append(rho_ac)
    if prop_a is not None:
        to_transform.append(prop_a * 2. - 1.)
    if prop_b is not None:
        to_transform.append(prop_b * 2. - 1.)
    if prop_c is not None:
        to_transform.append(prop_c * 2. - 1.)

    to_transform = tf.stack(to_transform, axis=1)
    return transform_correlations(to_transform)


def untransform_params(
        params_transformed: Tensor2[tf32, Batch, UnknownParams],
        param_is_proportion: Sequence[bool],
) -> Tuple[Tensor1[tf32, Batch], ...]:

    assert len(param_is_proportion) == params_transformed.shape[1]

    params_stretched = untransform_correlations(params_transformed)
    params_stretched = tf.unstack(params_stretched, axis=1)

    params = tuple(
        (value_tensor + 1.) / 2. if is_prop else value_tensor
        for value_tensor, is_prop in zip(params_stretched, param_is_proportion)
    )
    return params


def likelihoods_via_bfgs(
        log_likelihood_fn: Callable,
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],  # Stats = AB | ABC
        n: Tensor1[tf32, Batch],
        rho_ab_possibly_partial_null: Tensor1[tf32, Batch],
        rho_ab_possibly_partial_alt: Optional[Tensor1[tf32, Batch]],
        rho_bc: Optional[Tensor1[tf32, Batch]] = None,
        rho_ac: Optional[Tensor1[tf32, Batch]] = None,
        prop_a: Optional[Tensor1[tf32, Batch]] = None,
        prop_b: Optional[Tensor1[tf32, Batch]] = None,
        prop_c: Optional[Tensor1[tf32, Batch]] = None,
) -> Tensor1[tf32, Ys]:

    # We will even allow it to cheat by knowing the population params!!
    params_unknown_transformed_init = transform_params(
        rho_ab_possibly_partial_null,
        rho_bc,
        rho_ac,
        prop_a,
        prop_b,
        prop_c,
    )

    neg_log_likelihood_fn_free = functools.partial(
        lambda *args, **kw:
            -log_likelihood_from_transformed_params(*args, **kw),
        log_likelihood_fn, stats_tensor, n,
    )

    rho_ab_possibly_partial_null_trans = transform_correlations(
        rho_ab_possibly_partial_null,
    )
    neg_log_likelihood_fn_null = functools.partial(
        lambda *args, **kw:
            -log_likelihood_from_transformed_params_fixed_ab(*args, **kw),
        log_likelihood_fn, stats_tensor, n, rho_ab_possibly_partial_null_trans,
    )

    res_free = tfp.optimizer.bfgs_minimize(
        lambda params_unknown_transformed: tfp.math.value_and_gradient(
            neg_log_likelihood_fn_free,
            params_unknown_transformed,
        ),
        params_unknown_transformed_init,
        stopping_condition=tfp.optimizer.converged_all,
    )
    params_unknown_transformed_free = res_free.position
    likelihood_free = -neg_log_likelihood_fn_free(
        params_unknown_transformed_free,
        penalise_boundaries=False,
    )

    res_null = tfp.optimizer.bfgs_minimize(
        lambda params_unknown_transformed: tfp.math.value_and_gradient(
            neg_log_likelihood_fn_null,
            params_unknown_transformed,
        ),
        params_unknown_transformed_free[:, 1:] + 0.01,                         # Need to add a 0.01 here as, if the values passed in already converged, it will crash with NaNs!!
        stopping_condition=tfp.optimizer.converged_all,
    )
    likelihood_null = -neg_log_likelihood_fn_null(
        res_null.position,
        penalise_boundaries=False,
    )

    if rho_ab_possibly_partial_alt is not None:
        rho_ab_possibly_partial_alt_trans = transform_correlations(
            rho_ab_possibly_partial_alt,
        )
        neg_log_likelihood_fn_alt = functools.partial(
            lambda *args, **kw:
            -log_likelihood_from_transformed_params_fixed_ab(*args, **kw),
            log_likelihood_fn, stats_tensor, n,
            rho_ab_possibly_partial_alt_trans,
        )
        res_alt = tfp.optimizer.bfgs_minimize(
            lambda params_unknown_transformed: tfp.math.value_and_gradient(
                neg_log_likelihood_fn_alt,
                params_unknown_transformed,
            ),
            params_unknown_transformed_free[:, 1:] + 0.01,                     # Need to add a 0.01 here as, if the values passed in already converged, it will crash with NaNs!!
            stopping_condition=tfp.optimizer.converged_all,
        )
        likelihood_alt = -neg_log_likelihood_fn_alt(
            res_alt.position,
            penalise_boundaries=False,
        )
    else:
        likelihood_alt = likelihood_null
        res_alt = res_null

    float32 = tf.float32
    all_same_list = []
    for i, prop_i in enumerate((prop_a, prop_b, prop_c)):
        if prop_i is not None:
            num_true_i = tf.reduce_sum(stats_tensor[:, :, i], axis=1)
            all_same = tf.cast((num_true_i == 0.) | (num_true_i == n), float32)
            all_same_list.append(all_same)
    all_same_per_i = tf.stack(all_same_list, axis=1)
    all_same_count = tf.reduce_sum(all_same_per_i, axis=1)

    not_converged_free = tf.cast(~res_free.converged, float32)
    not_converged_null = tf.cast(~res_null.converged, float32)

    failed_free = tf.cast(res_free.failed, float32)
    failed_null = tf.cast(res_null.failed, float32)
    if rho_ab_possibly_partial_alt is not None:
        not_converged_alt = tf.cast(~res_alt.converged, float32)
        failed_alt = tf.cast(res_alt.failed, float32)
    else:
        not_converged_alt = tf.zeros_like(not_converged_null, float32)
        failed_alt = tf.zeros_like(failed_null, float32)

    failure_code = (not_converged_free
                    + not_converged_null * 2.
                    + not_converged_alt * 4.
                    + failed_free * 8.
                    + failed_null * 16.
                    + failed_alt * 32.
                    + all_same_count * 64.)

    likelihoods = tf.stack([likelihood_free,
                            likelihood_null,
                            likelihood_alt], axis=1)

    results = tf.concat([likelihoods,
                         failure_code[:, None],
                         res_free.position,
                         res_null.position,
                         res_alt.position], axis=1)

    return results
