import os

import biparcorr_analyse_funcs as biparcorr
import biparcorr_likelihood_ratio_funcs as lr

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def run_bfgs_likelihoods(
        method_name: str,
        num_sims_per_param_sample: int,
        start_from_param_num: int = 0,
        batch_size: int = 500,
        use_power_param_as_null: bool = False,  # for idealized power
) -> None:

    params = biparcorr.load_params_dict()
    num_param_samples = biparcorr.get_num_param_samples(params)
    params.pop('target_power')

    assert num_sims_per_param_sample % batch_size == 0

    # Do an initial run to force a compile so that our timings are pure
    print("Compiling!")
    lr.likelihoods_for_batch(
        simulation_block_num=tf.constant(0, tf.int32),
        first_sim_num=tf.constant(0, tf.int32),
        num_sims=batch_size,
        is_powersim_run=use_power_param_as_null,
        is_bootstrap_simulation=False,
        rho_ab_partial_null=tf.constant(0., tf.float32),
        rho_ab_partial_alt=tf.constant(0., tf.float32),
        rho_bc=tf.constant(0., tf.float32),
        rho_ac=tf.constant(0., tf.float32),
        prop_a=tf.constant(0.5, tf.float32),
        n=tf.constant(50., tf.float32),
    )
    print("Done compiling.")

    batch_size_tf = tf.constant(batch_size)
    for params_sample_num in tqdm(
            range(start_from_param_num, num_param_samples),
    ):
        batch_likelihoods = []
        params_sample_num_tf = tf.constant(params_sample_num, tf.int32)

        this_params = {n: p[params_sample_num] for n, p in params.items()}
        if use_power_param_as_null:
            rho_ab_partial_null = this_params.pop("rho_ab_partial_power")
            rho_ab_partial_alt = this_params.pop("rho_ab_partial")
        else:
            rho_ab_partial_null = this_params.pop("rho_ab_partial")
            rho_ab_partial_alt = this_params.pop("rho_ab_partial_power")

        for batch_num in range(num_sims_per_param_sample // batch_size):
            this_likelihoods = lr.likelihoods_for_batch(
                simulation_block_num=params_sample_num_tf,
                first_sim_num=batch_size_tf * batch_num,
                num_sims=batch_size,
                is_powersim_run=use_power_param_as_null,
                is_bootstrap_simulation=False,
                rho_ab_partial_null=rho_ab_partial_null,
                rho_ab_partial_alt=rho_ab_partial_alt,
                **this_params,
            )
            batch_likelihoods.append(this_likelihoods)

        filename = biparcorr.data_filename(method_name, "likelihoods",
                                           params_sample_num)
        path = biparcorr.convert_relative_path(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        likelihoods = tf.concat(batch_likelihoods, axis=0)
        np.save(path, likelihoods.numpy())


if __name__ == "__main__":
    run_bfgs_likelihoods("bfgs", 1_000_000)
    run_bfgs_likelihoods("bfgs_powersim", 10_000, use_power_param_as_null=True)
