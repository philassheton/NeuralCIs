import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from examples import comparisons_funcs
from examples import latent_correlation_funcs as lcor

# typing
from typing import Optional, Dict, Tuple
from collections.abc import Sequence
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor3
from tensor_annotations.tensorflow import float32 as tf32, int32 as ti32
from examples import Batch, Samples, Stats


COMPARISON_NAME = 'biserial_correlation'
PARAM_NAMES = ['rho_ab', 'prop_a', 'n', 'rho_ab_power', 'target_power']
PARAMS_BOUNDS_CHECKS = {
    "abs_rho_above_988": (lambda x: np.abs(x) >= np.atanh(0.988),
                          [True, False]),
    "prop_a_trans_abs_above_988": (lambda x: np.abs(x) >= np.atanh(0.988),
                                   [False, True]),
}


def sampling_distribution_fn_raw(
        rho_ab: Tensor1[tf32, Batch],
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

    return lcor.sampling_distribution_fn_raw_generic(
        rho_ab=rho_ab,
        prop_a_b_c=(prop_a, None),
        n=n,
        simulation_block_num=simulation_block_num,
        num_rows=num_rows,
        batch_consistent_randoms=batch_consistent_randoms,
        first_row_num_within_simulation_block=
                                        first_row_num_within_simulation_block,
        is_powersim_run=is_powersim_run,
        is_bootstrap_simulation=is_bootstrap_simulation,
    )


###############################################################################
#  Likelihood ratio functions
###############################################################################


def log_likelihood(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        rho_ab: Tensor1[tf32, Batch],
        prop_a: Tensor1[tf32, Batch],
        n: Tensor1[tf32, Batch],
) -> Tensor1[tf32, Batch]:

    std_norm = tfp.distributions.Normal(0., 1.)
    z_threshold_a = -std_norm.quantile(prop_a)
    a, b = tf.unstack(stats_tensor, axis=2)  # [Batch, 100]
    log_prob_b = std_norm.log_prob(b)
    mean_a_given_b = b * rho_ab[:, None]
    var_a_given_b = 1. - tf.square(rho_ab)                                     # like 1 - beta^2 -- variance not explained by the regression (beta = rho, and is the factor applied to the SD=1 of B when predicting A from it)
    sd_a_given_b = tf.sqrt(var_a_given_b)
    threshold_a_given_b_as_z_score = (
            (z_threshold_a[:, None] - mean_a_given_b)
            / sd_a_given_b[:, None]
    )
    p_one = 1. - std_norm.cdf(threshold_a_given_b_as_z_score)
    p_one = tf.clip_by_value(p_one, 1e-7, 1. - 1e-7)

    one = tf.constant(1.)
    log_prob_a = a * tf.math.log(p_one) + (one - a) * tf.math.log(one - p_one)

    log_prob_ab_masked = lcor.n_mask(n) * (log_prob_a + log_prob_b)
    log_likelihood_final = tf.reduce_sum(log_prob_ab_masked, axis=1)

    return log_likelihood_final                                                # type: ignore


@tf.function(jit_compile=True)
def likelihoods_for_batch(
        simulation_block_num: Tensor0[ti32],
        first_sim_num: Tensor0[ti32],
        num_sims: int,
        is_powersim_run: bool,
        is_bootstrap_simulation: bool,
        rho_ab_null: Tensor0[tf32],
        rho_ab_alt: Optional[Tensor0[tf32]] = None,
        **other_params: Tensor0[tf32],
):
    print("Compiling likelihoods_for_batch")
    params = {"rho_ab_null": rho_ab_null,
              "rho_ab_alt": rho_ab_alt} | other_params
    params = {n: tf.repeat(p, num_sims) if p is not None else None
              for n, p in params.items()}

    stats = sampling_distribution_fn_raw(
        rho_ab=params["rho_ab_null"],
        prop_a=params["prop_a"],
        n=params["n"],

        simulation_block_num=simulation_block_num,
        batch_consistent_randoms=True,
        first_row_num_within_simulation_block=first_sim_num,
        num_rows=num_sims,
        is_powersim_run=is_powersim_run,
        is_bootstrap_simulation=is_bootstrap_simulation,
    )

    likelihoods = lcor.likelihoods_via_bfgs(
        rho_ab_possibly_partial_null=params["rho_ab_null"],
        rho_ab_possibly_partial_alt=params["rho_ab_alt"],#
        prop_a=params["prop_a"],
        n=params["n"],

        stats_tensor=stats,

        log_likelihood_fn=log_likelihood,
    )

    return likelihoods


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
        only_first_n_pvalues, PARAMS_BOUNDS_CHECKS,
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
