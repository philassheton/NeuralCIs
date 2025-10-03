import biparcorr_analyse_funcs as biparcorr
import biparcorr_likelihood_ratio_funcs as lr
from neuralcis import NeuralCIs

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import functools
from tqdm import tqdm

# typing
from neuralcis.common import Batch, Samples, Stats, Ys
from tensor_annotations.tensorflow import Tensor1, Tensor3
from tensor_annotations.tensorflow import float32 as tf32


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
        batch_size: int,
        **params_per_batch_sample: Tensor1[tf32, Batch],
):

    line_search_args = [
        params_per_batch_sample[key] for key in [
            'rho_ab_partial', 'rho_ab_partial_power',
            'rho_bc', 'rho_ac', 'prop_a', 'n'
        ]
    ]
    rho_ab_partial_power = params_per_batch_sample.pop('rho_ab_partial_power')
    stats = biparcorr.sampling_distribution_fn_raw(batch_size=batch_size,
                                                   **params_per_batch_sample)
    line_search_args += [stats]
    likelihoods = likelihoods_via_line_search(*line_search_args)

    return likelihoods


cis = NeuralCIs.load('saved_model', 'testing')
params = biparcorr.load_or_generate_params_dict(cis)
batch_size = 1000

num_param_samples = biparcorr.get_num_param_samples(params)
target_powers = params.pop('target_power')

likelihoods_list = []
for param_sample_num in tqdm(range(num_param_samples)):
    biparcorr.start_first_batch_for_param_sample(param_sample_num)
    params_repeated = biparcorr.replicate_params(params,
                                                 param_sample_num,
                                                 batch_size)
    this_likelihoods = likelihoods_for_batch(batch_size,
                                             **params_repeated)
    likelihoods_list.append(this_likelihoods)


likelihoods = tf.stack(likelihoods_list)

likelihoods_path = biparcorr.convert_relative_path('exact_likelihoods.npy')
np.save(likelihoods_path, likelihoods.numpy())
