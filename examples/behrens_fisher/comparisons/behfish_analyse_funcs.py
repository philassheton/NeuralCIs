import tensorflow as tf
import numpy as np

from neuralcis import comparisons_funcs

# typing
from typing import Optional, Dict, Tuple
from collections.abc import Sequence
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor3
from tensor_annotations.tensorflow import float32 as tf32, int32 as ti32

from neuralcis.comparisons_funcs import Samples, Two


COMPARISON_NAME = 'behrens_fisher'
PARAM_NAMES = ['mudiff', 'sigma1', 'sigma2', 'n1', 'n2',
               'mudiff_power', 'target_power']


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
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

    return comparisons_funcs.summarise_pvalues_files(
        PARAM_NAMES, COMPARISON_NAME,
        method_name, data_type, alphas, add_params, num_params,
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
