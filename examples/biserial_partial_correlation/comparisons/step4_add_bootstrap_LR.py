import os

import biparcorr_analyse_funcs as biparcorr
import biparcorr_likelihood_ratio_funcs as lr

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# typing
from biparcorr_analyse_funcs import Params
from typing import Tuple
from tensor_annotations.tensorflow import Tensor0, Tensor1
from tensor_annotations.tensorflow import float32 as tf32


MAX_SIMS_PER_PARAM_SAMPLE = 10000
MAX_PARAM_SAMPLES = 1000



def bootstrap_single_p(
        log_likelihood_ratio: Tensor0[tf32],

        pos_null_transformed: Tensor1[tf32, Params],                           # transformed positions of all params except rho_ab_partial and n
        rho_ab_partial: Tensor0[tf32],
        n: Tensor0[tf32],

        bootstrap_samples: int,
        batch_size: int,

        params_sample_num: int,
        p_num: int,

        is_power_p: bool,
        is_powersim_run: bool,
) -> Tuple[Tensor0[tf32], Tensor0[tf32], Tensor0[tf32]]:

    bootstrap_batch_likelihoods = []
    rho_bc_sim, rho_ac_sim, prop_a_sim = lr.untransform_params(
        pos_null_transformed[None, :],
    )

    # For unique random seeds
    simulation_block_num = tf.constant(
        params_sample_num * MAX_SIMS_PER_PARAM_SAMPLE + p_num,
        dtype=tf.int32,
    ) + tf.where(is_power_p, MAX_PARAM_SAMPLES * MAX_SIMS_PER_PARAM_SAMPLE, 0)
    batch_size_tf = tf.constant(batch_size, dtype=tf.int32)

    for batch_num in range(bootstrap_samples // batch_size):
        this_likelihoods = lr.likelihoods_for_batch(
            simulation_block_num=simulation_block_num,
            first_sim_num=batch_size_tf * batch_num,
            num_sims=batch_size,
            simulate_from_power_rho=is_powersim_run,
            is_bootstrap_simulation=True,
            rho_ab_partial=rho_ab_partial,
            rho_bc=rho_bc_sim[0],  # Is anyway a length-1 vector
            rho_ac=rho_ac_sim[0],
            prop_a=prop_a_sim[0],
            n=n,
        )
        bootstrap_batch_likelihoods.append(this_likelihoods)

    bootstrap_likelihoods = tf.concat(bootstrap_batch_likelihoods, axis=0)
    bootstrap_lrs = bootstrap_likelihoods[:, 0] - bootstrap_likelihoods[:, 1]
    p = (tf.reduce_sum(tf.cast(bootstrap_lrs > log_likelihood_ratio, tf.int32))
         / (bootstrap_samples + 1))

    failed = bootstrap_likelihoods[:, 3] % 64 > 0.
    all_a_same = bootstrap_likelihoods[:, 3] >= 64

    failed_prop = tf.reduce_mean(tf.cast(failed, tf.float32))
    all_a_same_prop = tf.reduce_mean(tf.cast(all_a_same), tf.float32)

    return p, failed_prop, all_a_same_prop


def bootstrap_bfgs_likelihoods(
        method_name: str,
        num_ps_per_param_sample: int = 500,
        num_param_samples: int = 500,
        bootstrap_samples: int = 2000,
        start_from_param_num: int = 0,
        batch_size: int = 500,
        is_powersim_run: bool = False,
) -> None:

    params = biparcorr.load_params_dict()
    num_param_samples_available = biparcorr.get_num_param_samples(params)
    assert num_param_samples <= num_param_samples_available
    params.pop('target_power')

    reference_method = "bfgs_powersim" if is_powersim_run else "bfgs"

    assert bootstrap_samples % batch_size == 0

    # Do an initial run to force a compile so that our timings are pure
    print("Compiling!")
    lr.likelihoods_for_batch(
        simulation_block_num=tf.constant(0, tf.int32),
        first_sim_num=tf.constant(0, tf.int32),
        num_sims=batch_size,
        simulate_from_power_rho=is_powersim_run,
        is_bootstrap_simulation=True,
        rho_ab_partial=tf.constant(0., tf.int32),
        rho_bc=tf.constant(0., tf.int32),
        rho_ac=tf.constant(0., tf.int32),
        prop_a=tf.constant(0., tf.int32),
        n=tf.constant(0., tf.int32),
    )
    print("Done compiling.")

    for params_sample_num in tqdm(range(start_from_param_num,
                                        num_param_samples)):

        filename = biparcorr.data_filename(reference_method, "likelihoods",
                                           params_sample_num)
        likelihoods_data = np.load(filename)[0:num_ps_per_param_sample]
        likelihoods_data = tf.constant(likelihoods_data)
        likelihoods, error_code, pos_alt, pos_null, pos_power_null = \
            tf.split(likelihoods_data, [3, 1, 4, 3, 3], axis=1)
        likelihoods_alt, likelihoods_null, likelihoods_power_null = \
            tf.unstack(likelihoods, axis=1)

        n = params["n"][params_sample_num]
        rho_ab_partial = params["rho_ab_partial"][params_sample_num]
        rho_ab_partial_pow = params["rho_ab_partial_power"][params_sample_num]

        ps = []
        failed = []
        all_a_same = []
        ps_power = []
        failed_power = []
        all_a_same_power = []
        for p_num in range(num_ps_per_param_sample):
            p, f, aas = bootstrap_single_p(
                likelihoods_alt[p_num] - likelihoods_null[p_num],
                pos_null[p_num, :], rho_ab_partial, n,
                bootstrap_samples, batch_size,
                params_sample_num, p_num,
                is_power_p=False, is_powersim_run=is_powersim_run,
            )
            ps.append(p)
            failed.append(f)
            all_a_same.append(aas)

            p, f, aas = bootstrap_single_p(
                likelihoods_alt[p_num] - likelihoods_power_null[p_num],
                pos_power_null[p_num, :], rho_ab_partial_pow, n,
                bootstrap_samples, batch_size,
                params_sample_num, p_num,
                is_power_p=True, is_powersim_run=is_powersim_run,
            )
            ps_power.append(p)
            failed_power.append(f)
            all_a_same_power.append(aas)

        ps = tf.stack(ps)
        ps_power = tf.stack(ps_power)

        failed = tf.stack(failed)
        failed_power = tf.stack(failed_power)

        all_a_same = tf.stack(all_a_same)
        all_a_same_power = tf.stack(all_a_same_power)

        ps = tf.stack([ps, ps_power,
                       failed, failed_power,
                       all_a_same, all_a_same_power], axis=1)

        filename = biparcorr.data_filename(method_name, "ps",
                                           params_sample_num)
        path = biparcorr.convert_relative_path(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, ps.numpy())


if __name__ == "__main__":
    bootstrap_bfgs_likelihoods("bootstrap_lr")
    bootstrap_bfgs_likelihoods("bootstrap_lr_powersim", is_powersim_run=True)
