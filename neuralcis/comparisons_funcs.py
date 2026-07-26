import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
from tqdm import tqdm

# typing
from typing import Optional, Dict, Tuple
from collections.abc import Sequence
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3
from tensor_annotations.tensorflow import float32 as tf32


# These could be imported from neuralcis, but for easier deployment on an
#   instance, I want to avoid needing those unnecessary dependencies
import typing
from tensor_annotations import axes
Batch = typing.NewType("Batch", axes.Axis)
Samples = typing.NewType("Samples", axes.Axis)
Stats = typing.NewType("Stats", axes.Axis)
UnknownParams = typing.NewType("UnknownParams", axes.Axis)
Params = typing.NewType("Params", axes.Axis)
Ys = typing.NewType("Ys", axes.Axis)
One = typing.NewType("One", axes.Axis)
Two = typing.NewType("Two", axes.Axis)
PairwiseCorrelations = typing.NewType("PairwiseCorrelations", axes.Axis)


PARAMS_DICT_FILENAME = 'param_samples.npy'


def replicate_params(
        params_dict: Dict[str, Tensor1[tf32, Samples]],
        param_sample_index: int,
        batch_size: int,
) -> Dict[str, Tensor1[tf32, Batch]]:

    return {n: tf.repeat(p[param_sample_index], batch_size)
            for n, p in params_dict.items()}

def __make_cdf_summaries(
        ps: Tensor2[tf32, Samples, Two],
) -> np.ndarray:

    full_summary_quantiles = tf.linspace(0.0005, 0.9995, 1000)
    left_tail_quantiles = tf.linspace(0.00005, 0.09995, 1000)

    full_summary = tf.keras.ops.quantile(ps, full_summary_quantiles, axis=0)
    left_summary = tf.keras.ops.quantile(ps, left_tail_quantiles, axis=0)

    summaries = tf.stack([full_summary, left_summary])

    return summaries


def __kl_div_vs_uniform_hist(
    ps: Tensor1[tf32, Samples],
    bins: int = 1000,
) -> Tensor0[tf32]:

    ps = tf.cast(ps, dtype=tf.float64)

    counts = tf.histogram_fixed_width(ps, (0., 1.), nbins=bins)
    counts = tf.cast(counts, dtype=tf.float64)

    # Smoothed bin probabilities (Dirichlet prior with 0.5 per bin)
    pseudocount = 0.5  # corresponds to Jeffreys prior, avoids infinities
    ps_bin = ((counts + pseudocount)
              / (tf.reduce_sum(counts) + pseudocount*bins))

    uniform_bin = 1.0 / bins

    kl = tf.reduce_sum(ps_bin * tf.math.log(ps_bin / uniform_bin))
    return tf.cast(kl, dtype=tf.float32)


def __ks_dist_vs_uniform_over_region(
        ps: Tensor1[tf32, Samples],
        left_n_proportion: float = 1.,
):

    n_total = len(ps)
    x = tf.concat([[0.], tf.sort(ps), [1.]], axis=0)
    y_ecdf = tf.linspace(0., 1., n_total + 1)

    if left_n_proportion < 1.:
        n_in_region = int(n_total * left_n_proportion)
        x = x[:n_in_region+2]
        y_ecdf = y_ecdf[:n_in_region+1]

    # The empirical CDF is a stepped function with
    #  - variable step widths, and
    #  - fixed step heights
    y_uniform_start_of_step = x[:-1]
    y_uniform_end_of_step = x[1:]

    amount_above_at_start_of_step = y_ecdf - y_uniform_start_of_step
    amount_below_at_end_of_step = y_uniform_end_of_step - y_ecdf
    ks_dist = tf.maximum(
        tf.reduce_max(amount_above_at_start_of_step),
        tf.reduce_max(amount_below_at_end_of_step),
    )
    return ks_dist


def __likelihoods_to_ps(
        likelihoods: np.ndarray,
) -> Tuple[np.ndarray, int, int]:

    diffs = np.stack([
        likelihoods[:, 0] - likelihoods[:, 1],
        likelihoods[:, 0] - likelihoods[:, 2],
    ], axis=1)
    num_negative, num_negative_power = (diffs<0).sum(0).tolist()
    diffs = np.maximum(diffs, 0.0)
    ps = tf.math.igammac(0.5, 0.5 * 2.0 * diffs).numpy()

    return ps, num_negative, num_negative_power


def __failure_proportions_from_likelihoods_file(
        params_sample_num: int = 0,
        method: str = "bfgs",
        data_type: str = "likelihoods",  # might also be likelihoods_pow
) -> Dict[str, float]:

    assert data_type in ("likelihoods", "likelihoods_pow")
    filename = data_filename(method, data_type, params_sample_num)
    data = np.load(filename)
    failure_code = data[:, 3]

    def next_failure(
            failure_code: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        next_failure_bool = (failure_code % 2).astype(bool)
        next_code_to_pass_back = np.floor(failure_code / 2)
        return next_failure_bool, next_code_to_pass_back

    failure_types = ["not_converged_alt",
                     "not_converged_null",
                     "not_converged_power_null",
                     "failed_alt",
                     "failed_null",
                     "failed_power_null",
                     "all_a_same"]

    code = failure_code
    failures = {}
    for ftype in failure_types:
        failures[ftype], code = next_failure(code)

    failure_proportions = {k: v.mean().item() for k, v in failures.items()}

    return failure_proportions


def __load_data_file_as_pvalues(
        params_sample_num: int = 0,
        method: str = "neural",
        data_type: str = "ps",
) -> Tuple[np.ndarray, Dict[str, float]]:

    filename = data_filename(method, data_type, params_sample_num)
    data = np.load(filename)
    if data_type == "likelihoods":
        ps, num_negative, num_negative_power = __likelihoods_to_ps(data)
        extra_results = {f"neg_likelihoods_{method}": num_negative,
                         f"neg_likelihoods_power_{method}": num_negative_power}
    elif data_type == "ps":
        ps = data
        extra_results = {}
    else:
        raise Exception(f"Unknown data_type {data_type}")

    assert np.isfinite(ps).all()

    return ps, extra_results


def __summarise_pvalues(
        ps: Tensor2[tf32, Samples, Two],
        ps_powersim: Tensor2[tf32, Samples, Two],
        method_name: str,
        alphas: Sequence[float],
) -> Dict[str, float]:

    # GLOBAL COMPARISONS WITH UNIFORM
    results = {
        f"ks_dist_{method_name}":
            __ks_dist_vs_uniform_over_region(ps[:, 0]),
        f"ks_dist_tail_0.10_{method_name}":
            __ks_dist_vs_uniform_over_region(ps[:, 0], left_n_proportion=0.10),
        f"ks_dist_tail_0.05_{method_name}":
            __ks_dist_vs_uniform_over_region(ps[:, 0], left_n_proportion=0.05),
        f"kl_div_{method_name}":
            __kl_div_vs_uniform_hist(ps[:, 0]),
    }

    # LOCAL COMPARISONS AT GIVEN ALPHAS
    for alpha in alphas:
        cutoff_null_dist = tfp.stats.percentile(ps[:, 0], alpha * 100)
        cutoff_power_dist = tfp.stats.percentile(ps_powersim[:, 1],
                                                 alpha * 100)

        ps_below_alpha = tf.cast(ps < alpha, tf.float32)
        ps_below_null = tf.cast(ps < cutoff_null_dist, tf.float32)
        ps_below_power = tf.cast(ps < cutoff_power_dist, tf.float32)

        prop_below_alpha = tf.reduce_mean(ps_below_alpha, axis=0)
        prop_below_null = tf.reduce_mean(ps_below_null, axis=0)
        prop_below_power = tf.reduce_mean(ps_below_power, axis=0)

        error_rate, power = tf.unstack(prop_below_alpha)
        error_rate_perfect, power_adj_null = tf.unstack(prop_below_null)
        error_rate_adj_power, power_perfect = tf.unstack(prop_below_power)

        suffix = f"{alpha:.3f}_{method_name}"
        results |= {
            f"error_rate_{suffix}": error_rate,
            f"power_{suffix}": power,
            f"error_rate_perfect_{suffix}": error_rate_perfect,
            f"power_adj_null_{suffix}": power_adj_null,
            f"error_rate_adj_power_{suffix}": error_rate_adj_power,
            f"power_perfect_{suffix}": power_perfect,
        }
    return results


# TODO: if we can swap tf.histogram_fixed_width for an XLA-compatible
#       equivalent, then we can also jit_compile.
@tf.function(jit_compile=False)
def __compute_summaries(
        ps: Tensor2[tf32, Samples, Two],
        ps_powersim: Tensor2[tf32, Samples, Two],
        method_name: str,
        alphas: Sequence[float],
) -> Tuple[Dict[str, Tensor0], Tensor3[tf32, Two, Samples, Two]]:

    main_results = __summarise_pvalues(ps, ps_powersim, method_name, alphas)
    summary_grids = __make_cdf_summaries(ps)
    return main_results, summary_grids


def __summarise_data_file(
        params_sample_num: int = 0,
        method_name: str = "neural",
        data_type: str = "ps",
        alphas: Sequence[float] = (0.05, 0.01),
        only_first_n_pvalues: Optional[int] = None,
) -> Tuple[Dict[str, float], np.ndarray]:

    ps, extra_results = __load_data_file_as_pvalues(params_sample_num,
                                                    method_name,
                                                    data_type)
    ps_powersim, _ = __load_data_file_as_pvalues(params_sample_num,
                                                 f"{method_name}_powersim",
                                                 data_type)

    if only_first_n_pvalues is not None:
        assert data_type == "ps"
        ps = ps[0:only_first_n_pvalues, :]
        ps_powersim = ps_powersim[0:only_first_n_pvalues, :]

    if data_type in ("likelihoods", "likelihoods_pow"):
        failure_proportions = __failure_proportions_from_likelihoods_file(
            params_sample_num,
            method_name,
            data_type,
        )
        extra_results |= failure_proportions

    ps = tf.constant(ps)
    ps_powersim = tf.constant(ps_powersim)

    main_results, summary_grids = __compute_summaries(ps, ps_powersim,
                                                      method_name, alphas)

    main_results = {n:v.numpy().item() for n, v in main_results.items()}
    summary_grids = summary_grids.numpy()

    return main_results | extra_results, summary_grids


def summarise_pvalues_files(
        param_names: Sequence[str],
        comparison_name: str,
        method_name: str = "neural",
        data_type: str = "ps",
        alphas: Sequence[float] = (0.05, 0.01),
        add_params: bool = False,
        num_params: Optional[int] = None,
        only_first_n_pvalues: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

    params = load_params_dict(param_names, comparison_name)
    if num_params is None:
        num_params = get_num_param_samples(params)

    summaries = [__summarise_data_file(i, method_name, data_type, alphas,
                                       only_first_n_pvalues)
                     for i in tqdm(range(num_params))]
    summary_dict = {n:np.stack([dict[n] for (dict, grid) in summaries], axis=0)
                    for n in summaries[0][0]}
    summary_grids = np.stack([grid for dict, grid in summaries])
    if add_params:
        summary_dict |= {n:p.numpy() for n, p in params.items()}

    return summary_dict, summary_grids


def load_params_dict(
        param_names: Sequence[str],
        comparison_name: str,
) -> Optional[Dict[str, Tensor1[tf32, Samples]]]:

    param_samples_file = convert_relative_path(PARAMS_DICT_FILENAME,
                                               comparison_name)

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
        param_names: Sequence[str],
        comparison_name: str,
) -> None:

    param_samples_file = convert_relative_path(PARAMS_DICT_FILENAME,
                                               comparison_name)
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


def data_filename(
        method_name: str,
        data_type: str,
        params_sample_num: int,
) -> str:

    return (f"data/{method_name}/"
            f"{method_name}_{data_type}_{params_sample_num:04d}.npy")


def convert_relative_path(basename: str, comparison_name: str) -> str:
    if os.getcwd().endswith(f'{comparison_name}/comparisons'):
        dirname = ''
    elif os.getcwd().endswith('NeuralCIs'):
        dirname = f'examples/{comparison_name}/comparisons'
    elif os.getcwd().lower().endswith(comparison_name):
        dirname = 'comparisons'
    else:
        dirname = ''

    return os.path.join(dirname, basename)


def get_scalar_params(
        params: Dict[str, Tensor1[tf32, Batch]]
) -> Dict[str, float]:

    if np.all([tf.reduce_min(p) == tf.reduce_max(p) for p in params.values()]):
        return {n: float(p[0]) for n, p in params.items()}
    else:
        raise Exception('There are multiple different param values per param!')
