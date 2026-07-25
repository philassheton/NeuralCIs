import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from neuralcis import comparisons_funcs

# typing
from typing import Optional, Dict, Tuple
from collections.abc import Sequence
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3
from tensor_annotations.tensorflow import float32 as tf32, int32 as ti32

from neuralcis.comparisons_funcs import (Batch, Samples, Params, UnknownParams,
                                         Ys, Stats, PairwiseCorrelations)


COMPARISON_NAME = 'biserial_partial_correlation'
PARAM_NAMES = ['rho_ab_partial', 'rho_bc', 'rho_ac', 'prop_a', 'n',
               'rho_ab_partial_power', 'target_power']

N_MAX = 100
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
        lambda row: tf.random.stateless_normal(row_shape,
                                               tf.stack([simulation_block_num, row])),
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


def sampling_distribution_fn_raw(
        rho_ab_partial: Tensor1[tf32, Batch],
        rho_bc: Tensor1[tf32, Batch],
        rho_ac: Tensor1[tf32, Batch],
        prop_a: Tensor1[tf32, Batch],
        n: Tensor1[tf32, Batch],
        simulation_block_num: Tensor0[ti32],
        num_rows: int,

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

    rho_ab = (rho_bc * rho_ac
              + rho_ab_partial * tf.sqrt((1. - tf.square(rho_bc)) *
                                         (1. - tf.square(rho_ac))))

    z_threshold_a = -tfp.distributions.Normal(0., 1.).quantile(prop_a)

    one = tf.ones_like(rho_ab)
    correlation_matrix = tf.stack([
        tf.stack([one, rho_ab, rho_ac], axis=-1),
        tf.stack([rho_ab, one, rho_bc], axis=-1),
        tf.stack([rho_ac, rho_bc, one], axis=-1),
    ], axis=-1)

    cholesky = tf.linalg.cholesky(correlation_matrix)

    mask = n_mask(n)[:, None, :]
    z = generate_random_normals(num_rows,
                                (3, N_MAX),
                                simulation_block_num,
                                batch_consistent_randoms,
                                first_row_num_within_simulation_block) * mask
    z_correlated = tf.linalg.matmul(cholesky, z)

    a, b, c = tf.split(z_correlated, 3, axis=1)
    a = tf.cast(a > z_threshold_a[:, None, None], tf.float32) * mask
    samples = tf.stack([a[:, 0, :], b[:, 0, :], c[:, 0, :]], axis=2)

    return samples


# n_mask ensures we only have n z values (the rest will be zeroed)
#  -- this means we work with constant memory size.
def n_mask(n: Tensor1[tf32, Batch]) -> Tensor2[tf32, Batch, Samples]:
    n = tf.cast(n, tf.int64)
    return tf.cast(tf.sequence_mask(n, N_MAX), tf.float32)


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


###############################################################################
#  Stubs to hook into comparison_funcs
###############################################################################

def load_params_dict(
) -> Optional[Dict[str, Tensor1[tf32, Samples]]]:

    return comparisons_funcs.load_params_dict(PARAM_NAMES, COMPARISON_NAME)


def save_params_dict(
        params: Dict[str, Tensor1[tf32, Samples]],
) -> None:

    comparisons_funcs.save_params_dict(params, PARAM_NAMES, COMPARISON_NAME)


def summarise_pvalues_files(
        method_name: str = "neural",
        data_type: str = "ps",
        alphas: Sequence[float] = (0.05, 0.01),
        add_params: bool = False,
        num_params: Optional[int] = None,
        only_first_n_pvalues: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

    return comparisons_funcs.summarise_pvalues_files(
        PARAM_NAMES, COMPARISON_NAME,
        method_name, data_type, alphas, add_params, num_params,
        only_first_n_pvalues,
    )


def convert_relative_path(basename: str) -> str:
    return comparisons_funcs.convert_relative_path(basename, COMPARISON_NAME)


def get_num_param_samples(
        params_dict: Dict[str, Tensor1[tf32, Samples]],
) -> int:

    return comparisons_funcs.get_num_param_samples(params_dict)


def data_filename(
        method_name: str,
        data_type: str,
        params_sample_num: int,
) -> str:

    return comparisons_funcs.data_filename(method_name, data_type,
                                           params_sample_num)
