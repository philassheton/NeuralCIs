import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

# typing
from typing import Optional, Dict, Tuple
from collections.abc import Sequence
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor3
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
Two = typing.NewType("Two", axes.Axis)
PairwiseCorrelations = typing.NewType("PairwiseCorrelations", axes.Axis)


N_MAX = 100
PARAMS_DICT_FILENAME = 'param_samples.npy'


def __generate_random_normals(
        row_length: int,
        params_num: Tensor0[ti32],
        num_rows: int,
) -> Tensor3:

    return tf.random.stateless_normal((num_rows, row_length),
                                      tf.stack([params_num, 0]))


def __generate_random_chi2(
        row_length: int,
        df: Tensor1[tf32, Two],
        params_num: Tensor0[ti32],
        num_rows: int,
) -> Tensor3:

    alpha = df * 0.5
    beta = tf.constant(0.5, dtype=tf.float32)
    params_seed = params_num + 1_000_000  # Make sure is different from normal
    return tf.random.stateless_gamma((num_rows, row_length),
                                      tf.stack([params_seed, 0]),
                                      alpha=alpha,
                                      beta=beta)


def sampling_distribution_fn_stateless(
        mudiff: Tensor0[tf32],
        sigma1: Tensor0[tf32],
        sigma2: Tensor0[tf32],
        n1: Tensor0[tf32],
        n2: Tensor0[tf32],

        params_num: Tensor0[ti32],
        num_rows: int,
        seed_differently: bool,
) -> Dict[str, Tensor1[tf32, Samples]]:

    if seed_differently:
         params_num += 1_000_000_000

    df1 = n1 - 1.
    df2 = n2 - 1.
    dfs = tf.stack([df1, df2])

    zs = __generate_random_normals(2, params_num, num_rows)
    chi2s = __generate_random_chi2(2, dfs, params_num, num_rows)

    z1, z2 = tf.unstack(zs, axis=1)
    chi_sq1, chi_sq2 = tf.unstack(chi2s, axis=1)

    mu1_hat = z1 * sigma1 / tf.math.sqrt(n1)
    mu2_hat = z2 * sigma2 / tf.math.sqrt(n2) + mudiff
    mudiff_hat = mu2_hat - mu1_hat

    sigma1_hat = sigma1 * tf.math.sqrt(chi_sq1 / df1)
    sigma2_hat = sigma2 * tf.math.sqrt(chi_sq2 / df2)

    return {"mudiff_hat": mudiff_hat,
            "sigma1_hat": sigma1_hat,
            "sigma2_hat": sigma2_hat}


def replicate_params(
        params_dict: Dict[str, Tensor1[tf32, Samples]],
        param_sample_index: int,
        batch_size: int,
) -> Dict[str, Tensor1[tf32, Batch]]:

    return {n: tf.repeat(p[param_sample_index], batch_size)
            for n, p in params_dict.items()}


def __make_cdf_summaries(
        ps: np.ndarray,
) -> np.ndarray:

    full_summary_quantiles = np.linspace(0., 1., 1000, endpoint=False)
    left_tail_quantiles = np.linspace(0., 0.1, 1000, endpoint=False)

    full_summary_quantiles += full_summary_quantiles[1] / 2
    left_tail_quantiles += left_tail_quantiles[1] / 2

    full_summary = np.quantile(ps, full_summary_quantiles, axis=0)
    left_summary = np.quantile(ps, left_tail_quantiles, axis=0)

    summaries = np.stack([full_summary, left_summary])

    return summaries


def __kl_div_vs_uniform_hist(
    ps: np.ndarray,
    bins: int = 1000,
) -> float:

    ps = np.asarray(ps, dtype=np.float64)
    if (ps < 0).any() or (ps > 1).any():
        raise ValueError("p-values must be in [0,1].")

    counts, _ = np.histogram(ps, bins=bins, range=(0.0, 1.0))
    counts = counts.astype(np.float64)

    # Smoothed bin probabilities (Dirichlet prior with 0.5 per bin)
    pseudocount = 0.5  # corresponds to Jeffreys prior, avoids infinities
    ps_bin = (counts + pseudocount) / (counts.sum() + pseudocount*bins)

    uniform_bin = 1.0 / bins

    kl = np.sum(ps_bin * np.log(ps_bin / uniform_bin))
    return float(kl)


def __ks_dist_vs_uniform_over_region(
        ps: np.ndarray,
        right_p_boundary: float = 1.,
):

    n_total = len(ps)
    ps = ps[ps < right_p_boundary]
    n_in_region = len(ps)

    # The empirical CDF is a stepped function with
    #  - variable step widths, and
    #  - fixed step heights
    x = np.concatenate([[0], np.sort(ps), [right_p_boundary]])
    y_ecdf = np.linspace(0, 1, n_total + 1)[:n_in_region+1]
    y_uniform_start_of_step = x[:-1]
    y_uniform_end_of_step = x[1:]

    amount_above_at_start_of_step = y_ecdf - y_uniform_start_of_step
    amount_below_at_end_of_step = y_uniform_end_of_step - y_ecdf
    ks_dist = np.maximum(
        amount_above_at_start_of_step.max(),
        amount_below_at_end_of_step.max(),
    )

    return ks_dist.item()


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


def __likelihoodspow_to_ps(
        likelihoods: np.ndarray,
        likelihoodspow: np.ndarray,
) -> Tuple[np.ndarray, int, int]:

    # This is a short-cut for the case where power targeting was redesigned
    # and only power values were to be resampled.  Special extra files called
    # "likelihoodspow" instead of "likelihoods" store only the likelihoods
    # for the power component (having been initialised using the original
    # fit of the non-power null, extracted from the original likelihoods
    # file).
    #
    # This function merges the p-values from the new file into the old one.

    likelihoods[:, 2] = likelihoodspow[:, 0]
    return __likelihoods_to_ps(likelihoods)


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
    elif data_type == "likelihoodspow":
        filename_main = filename.replace("likelihoodspow", "likelihoods")
        data_main = np.load(filename_main)
        ps, num_negative, num_negative_power = __likelihoodspow_to_ps(data_main, data)  # PHIL!!  Hacks!!
        extra_results = {f"neg_likelihoods_{method}": num_negative,
                         f"neg_likelihoods_power_{method}": num_negative_power}
    elif data_type == "ps":
        ps = data
        extra_results = {}
    else:
        raise Exception(f"Unknown method {method}")

    return ps, extra_results


def __summarise_pvalues(
        ps: np.ndarray,
        method_name: str,
        alphas: Sequence[float],
) -> Dict[str, float]:

    # GLOBAL COMPARISONS WITH UNIFORM
    results = {
        f"ks_dist_{method_name}":
            __ks_dist_vs_uniform_over_region(ps[:, 0]),
        f"ks_dist_tail_{method_name}":
            __ks_dist_vs_uniform_over_region(ps[:, 0], right_p_boundary=0.1),
        f"kl_div_{method_name}":
            __kl_div_vs_uniform_hist(ps[:, 0]),
    }

    # LOCAL COMPARISONS AT GIVEN ALPHAS
    for alpha in alphas:
        cutoff_null_dist = np.percentile(ps[:, 0],
                                         alpha * 100)

        error_rate, power = (ps < alpha).mean(0).tolist()
        error_rate_perfect, power_adj_null = \
            (ps < cutoff_null_dist).mean(0).tolist()
        suffix = f"{alpha:.3f}_{method_name}"
        results |= {
            f"error_rate_{suffix}": error_rate,
            f"power_{suffix}": power,
            f"error_rate_perfect_{suffix}": error_rate_perfect,
            f"power_adj_null_{suffix}": power_adj_null,
        }
    return results


def __summarise_data_file(
        params_sample_num: int = 0,
        method_name: str = "neural",
        data_type: str = "ps",
        alphas: Sequence[float] = (0.05, 0.01),
) -> Tuple[Dict[str, float], np.ndarray]:

    ps, extra_results = __load_data_file_as_pvalues(params_sample_num,
                                                    method_name,
                                                    data_type)
    main_results = __summarise_pvalues(ps, method_name, alphas)

    summary_grids = __make_cdf_summaries(ps)

    return main_results | extra_results, summary_grids


def summarise_pvalues_files(
        method_name: str = "neural",
        data_type: str = "ps",
        alphas: Sequence[float] = (0.05, 0.01),
        add_params: bool = False,
        num_params: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

    params = load_params_dict()
    if num_params is None:
        num_params = get_num_param_samples(params)

    summaries = [__summarise_data_file(i, method_name, data_type, alphas)
                     for i in tqdm(range(num_params))]
    summary_dict = {n:np.stack([dict[n] for (dict, grid) in summaries], axis=0)
                    for n in summaries[0][0]}
    summary_grids = np.stack([grid for dict, grid in summaries])
    if add_params:
        summary_dict |= {n:p.numpy() for n, p in params.items()}

    return summary_dict, summary_grids


param_names = ['mudiff', 'sigma1', 'sigma2', 'n1', 'n2',
               'mudiff_power', 'target_power']


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


def data_filename(
        method_name: str,
        data_type: str,
        params_sample_num: int,
) -> str:

    if method_name == "neural elu":
        base = "/media/phil/Shared/neuralcis/behrens_fisher_files/data_elu"
    elif method_name == "neural":
        base = "data"
    elif method_name == "welch":
        base = "/media/phil/Shared/neuralcis/behrens_fisher_files/data"
    else:
        raise Exception(f"What method is {method_name}?")
    return f"{base}/{method_name}_{data_type}_{params_sample_num:04d}.npy"


def convert_relative_path(basename: str) -> str:
    if os.getcwd().endswith('behrens_fisher/comparisons'):
        dirname = ''
    elif os.getcwd().endswith('NeuralCIs'):
        dirname = 'examples/behrens_fisher/comparisons'
    elif os.getcwd().lower().endswith('behrens_fisher'):
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
