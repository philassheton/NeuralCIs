import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from datetime import datetime
import os

# typing
from typing import Optional, Dict, Tuple
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3
from tensor_annotations.tensorflow import float32 as tf32, int32 as ti32


# These could be imported from neuralcis, but for easier deployment on an
#   instance, I want to avoid needing those unnecessary dependencies
import typing
from tensor_annotations import axes
Batch = typing.NewType("Batch", axes.Axis)
Samples = typing.NewType("Samples", axes.Axis)
Stats = typing.NewType("Stats", axes.Axis)
UnknownParams = typing.NewType("UnknownParams", axes.Axis)
Ys = typing.NewType("Ys", axes.Axis)
One = typing.NewType("One", axes.Axis)
PairwiseCorrelations = typing.NewType("PairwiseCorrelations", axes.Axis)


N_MAX = 100
PARAMS_DICT_FILENAME = 'param_samples.npy'


def generate_random_normals(
        row_shape: Tuple[int, int],
        params_num: Tensor0[ti32],
        first_row_num: Tensor0[ti32],
        num_rows: int,
) -> Tensor3:

    row_nums = tf.range(num_rows) + first_row_num
    return tf.map_fn(
        lambda row: tf.random.stateless_normal(row_shape,
                                               tf.stack([params_num, row])),
        row_nums,
        dtype=tf.float32,
    )


def sampling_distribution_fn_raw(
        rho_ab_partial: Tensor1[tf32, Batch],
        rho_bc: Tensor1[tf32, Batch],
        rho_ac: Tensor1[tf32, Batch],
        prop_a: Tensor1[tf32, Batch],
        n: Tensor1[tf32, Batch],
        params_num: Tensor0[ti32],
        first_row_num: Tensor0[ti32],
        num_rows: int,
        seed_differently: bool,
) -> Tensor3[tf32, Batch, Samples, Stats]:

    if seed_differently:
        params_num += 1_000_000_000

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
    z = generate_random_normals((3, N_MAX),
                                params_num,
                                first_row_num,
                                num_rows) * mask
    z_correlated = tf.linalg.matmul(cholesky, z)

    a, b, c = tf.split(z_correlated, 3, axis=1)
    a = tf.cast(a > z_threshold_a[:, None, None], tf.float32) * mask
    samples = tf.stack([a[:, 0, :], b[:, 0, :], c[:, 0, :]], axis=2)

    return samples


def replicate_params(
        params_dict: Dict[str, Tensor1[tf32, Samples]],
        param_sample_index: int,
        batch_size: int,
) -> Dict[str, Tensor1[tf32, Batch]]:

    return {n: tf.repeat(p[param_sample_index], batch_size)
            for n, p in params_dict.items()}


# n_mask ensures we only have n z values (the rest will be zeroed)
#  -- this means we work with constant memory size.
def n_mask(n: Tensor1[tf32, Batch]) -> Tensor1[tf32, Samples]:
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


def make_cdf_summary(
        ps: Tensor2[tf32, Samples, Stats],
        summary_length: int,
) -> Tensor3[tf32, One, Stats, Samples]:

    ps_sorted = tf.sort(tf.transpose(ps), axis=1)
    end_of_bucket_index = tf.linspace(0., len(ps), summary_length + 1)[1:] - 1
    end_of_bucket_index = tf.cast(end_of_bucket_index, tf.int32)
    return tf.gather(ps_sorted, end_of_bucket_index, axis=1)[None, :, :]


param_names = ['rho_ab_partial', 'rho_bc', 'rho_ac', 'prop_a', 'n',
               'rho_ab_partial_power', 'target_power']


def rho_for_power(
        rho_null,
        n,
        signed_target_power,
        alpha: float = 0.05,
):
    normal = tfp.distributions.Normal(0., 1.)

    target_power = tf.math.abs(signed_target_power)
    alt_sign = tf.sign(signed_target_power)
    alpha = tf.constant(alpha, dtype=tf.float32)
    num_control_vars = tf.constant(1, dtype=tf.float32)

    z_alpha = normal.quantile(1.0 - alpha / 2.0)
    z_beta = normal.quantile(target_power)
    delta_z = (z_alpha + z_beta) / tf.sqrt(n - num_control_vars - 3.0)
    z_null = tf.atanh(tf.clip_by_value(rho_null, -0.999, 0.999))

    z_alt = z_null + alt_sign * delta_z
    rho_alt = tf.tanh(z_alt)

    return rho_alt


def load_params_dict(
) -> Optional[Dict[str, Tensor1[tf32, Samples]]]:

    param_samples_file = convert_relative_path(PARAMS_DICT_FILENAME)

    if os.path.exists(param_samples_file):
        params_grid = tf.convert_to_tensor(np.load(param_samples_file),
                                           dtype=tf.float32)
        param_tensors = tf.unstack(params_grid, axis=1)
        params = {name: param for name, param in
                  zip(param_names, param_tensors)}
        return params

    else:
        return None


def save_params_dict(
        params: Dict[str, Tensor1[tf32, Samples]],
) -> None:

    param_samples_file = convert_relative_path(PARAMS_DICT_FILENAME)
    param_tensors = [params[name] for name in param_names]
    params_grid = tf.stack(param_tensors, axis=1)
    np.save(param_samples_file, params_grid.numpy())


def get_num_param_samples(
        params_dict: Dict[str, Tensor1[tf32, Samples]],
) -> int:

    num_param_samples = np.unique([len(p) for p in params_dict.values()])
    if len(num_param_samples) != 1:
        raise Exception('Every param must be an equal length 1D Tensor!!')
    num_param_samples = int(num_param_samples[0])
    return num_param_samples


def convert_relative_path(basename: str) -> str:
    if os.getcwd().endswith('biserial_partial_correlation'):
        dirname = ''
    elif os.getcwd().lower().endswith('NeuralCIs'):
        dirname = 'examples/biserial_partial_correlation'
    else:
        raise Exception('What directory are we in??')

    return os.path.join(dirname, basename)


def get_scalar_params(
        params: Dict[str, Tensor1[tf32, Batch]]
) -> Dict[str, float]:

    if np.all([tf.reduce_min(p) == tf.reduce_max(p) for p in params.values()]):
        return {n: float(p[0]) for n, p in params.items()}
    else:
        raise Exception('There are multiple different param values per param!')


def param_run_filename(
        save_directory: str,
        layer_order_summary: str,
        params_index: int,
        params: Dict[str, Tensor1[tf32, Batch]],
        num_simulations_per_param: int,
) -> str:

    params = get_scalar_params(params)
    power_percent = int(params["target_power"] * 100)
    return (f'{save_directory}/'
            f'pars{params_index}'
            f' {layer_order_summary}'
            f' {datetime.now().strftime("%Y%m%d %H%M%S")}'
            f' r_ab_p {params["rho_ab_partial"]:.4f}'
            f' r_ab_p{power_percent:d}'
            f' {params["rho_ab_partial_power"]:.4f}'
            f' r_bc {params["rho_bc"]:.4f}'
            f' r_ac {params["rho_ac"]:.4f}'
            f' p_a {params["prop_a"]:.4f}'
            f' n {int(params["n"]):d}'
            f' runs {num_simulations_per_param}')
