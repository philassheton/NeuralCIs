import biparcorr_analyse_funcs as biparcorr
import biparcorr_likelihood_ratio_funcs as lr

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import functools
from tqdm import tqdm

# typing
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor3
from tensor_annotations.tensorflow import float32 as tf32, int32 as ti32
from biparcorr_analyse_funcs import Batch, Samples, Stats, Ys


def likelihoods_via_line_search(
        rho_ab_partial_null: Tensor1[tf32, Batch],
        rho_ab_partial_power: Tensor1[tf32, Batch],
        rho_bc: Tensor1[tf32, Batch],
        rho_ac: Tensor1[tf32, Batch],
        prop_a: Tensor1[tf32, Batch],
        n: Tensor1[tf32, Batch],
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
) -> Tensor1[tf32, Ys]:

    # We will even allow it to cheat by knowing the population params!!
    params_unknown_transformed_init = lr.transform_params(
        rho_ab_partial_null,
        rho_bc,
        rho_ac,
        prop_a,
    )
    rho_ab_partial_null_trans = lr.transform_correlations(rho_ab_partial_null)
    rho_ab_partial_pow_trans = lr.transform_correlations(rho_ab_partial_power)

    neg_log_likelihood_fn_alternative = functools.partial(
        lambda *args, **kw:
            -lr.log_likelihood_from_transformed_params(*args, **kw),
        stats_tensor, n,
    )
    neg_log_likelihood_fn_null = functools.partial(
        lambda *args, **kw:
            -lr.log_likelihood_from_transformed_params_fixed_ab(*args, **kw),
        stats_tensor, n, rho_ab_partial_null_trans,
    )
    neg_log_likelihood_fn_power_null = functools.partial(
        lambda *args, **kw:
            -lr.log_likelihood_from_transformed_params_fixed_ab(*args, **kw),
        stats_tensor, n, rho_ab_partial_pow_trans,
    )

    res_alt = tfp.optimizer.bfgs_minimize(
        lambda params_unknown_transformed: tfp.math.value_and_gradient(
            neg_log_likelihood_fn_alternative,
            params_unknown_transformed,
        ),
        params_unknown_transformed_init,
        stopping_condition=tfp.optimizer.converged_all,
    )
    params_unknown_transformed_alt = res_alt.position

    res_alt = tfp.optimizer.bfgs_minimize(
        lambda params_unknown_transformed: tfp.math.value_and_gradient(
            neg_log_likelihood_fn_alternative,
            params_unknown_transformed,
        ),
        params_unknown_transformed_alt,
        stopping_condition=tfp.optimizer.converged_all,
    )
    likelihood_alt = -neg_log_likelihood_fn_alternative(
        res_alt.position,
        penalise_boundaries=False,
    )

    res_null = tfp.optimizer.bfgs_minimize(
        lambda params_unknown_transformed: tfp.math.value_and_gradient(
            neg_log_likelihood_fn_null,
            params_unknown_transformed,
        ),
        params_unknown_transformed_alt[:, 1:],
        stopping_condition=tfp.optimizer.converged_all,
    )
    likelihood_null = -neg_log_likelihood_fn_null(
        res_null.position,
        penalise_boundaries=False,
    )

    res_power_null = tfp.optimizer.bfgs_minimize(
        lambda params_unknown_transformed: tfp.math.value_and_gradient(
            neg_log_likelihood_fn_power_null,
            params_unknown_transformed,
        ),
        params_unknown_transformed_alt[:, 1:],
        stopping_condition=tfp.optimizer.converged_all,
    )
    likelihood_power_null = -neg_log_likelihood_fn_power_null(
        res_power_null.position,
        penalise_boundaries=False,
    )

    num_true_a = tf.reduce_sum(stats_tensor[:, :, 0], axis=1)
    ill_conditioned = tf.cast((num_true_a == 0.) | (num_true_a == n),
                              tf.float32)

    likelihoods = tf.stack([likelihood_alt,
                            likelihood_null,
                            likelihood_power_null], axis=1)

    results = tf.concat([likelihoods,
                         ill_conditioned[:, None],
                         res_alt.position,
                         res_null.position,
                         res_power_null.position], axis=1)

    return results


@tf.function
def likelihoods_for_batch(
        params_num: Tensor0[ti32],
        first_sim_num: Tensor0[ti32],
        num_sims: int,
        **params: Tensor0[tf32],
):

    params = {n: tf.repeat(p, num_sims) for n, p in params.items()}
    line_search_args = [
        params[key] for key in [
            'rho_ab_partial', 'rho_ab_partial_power',
            'rho_bc', 'rho_ac', 'prop_a', 'n'
        ]
    ]
    params.pop('rho_ab_partial_power')
    stats = biparcorr.sampling_distribution_fn_raw(params_num=params_num,
                                                   first_row_num=first_sim_num,
                                                   num_rows=num_sims,
                                                   **params)
    line_search_args += [stats]
    likelihoods = likelihoods_via_line_search(*line_search_args)

    return likelihoods


cis = NeuralCIs.load('saved_model', 'testing')
params = biparcorr.load_or_generate_params_dict(cis)
batch_size = 500
start_from_param_num = 0
num_sims_per_param_sample = 1000
num_param_samples = biparcorr.get_num_param_samples(params)
params.pop('target_power')

assert num_sims_per_param_sample % batch_size == 0

batch_size_tf = tf.constant(batch_size)
for params_sample_num in tqdm(range(start_from_param_num, num_param_samples)):
    batch_likelihoods = []
    params_sample_num_tf = tf.constant(params_sample_num, tf.int32)
    # TODO: All this shit used to be nicely factored into biparcorr lib!!
    this_params = {n: p[params_sample_num] for n, p in params.items()}
    for batch_num in range(num_sims_per_param_sample // batch_size):
        this_likelihoods = likelihoods_for_batch(params_sample_num_tf,
                                                 batch_size_tf * batch_num,
                                                 batch_size,
                                                 **this_params)
        batch_likelihoods.append(this_likelihoods)

    filename = f'exact_likelihoods_{params_sample_num:04d}.npy'
    path = biparcorr.convert_relative_path(filename)
    likelihoods = tf.concat(batch_likelihoods, axis=0)
    np.save(path, likelihoods.numpy())
