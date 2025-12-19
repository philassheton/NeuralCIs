import biparcorr_analyse_funcs as biparcorr
from neuralcis import NeuralCIs

import numpy as np
import tensorflow as tf
import os

from tqdm import tqdm

# typing
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor3
from tensor_annotations.tensorflow import float32 as tf32, int32 as ti32


@tf.function(jit_compile=True)
def pvalues_for_batch(
        cis: NeuralCIs,
        params_num: Tensor0[ti32],
        first_sim_num: Tensor0[ti32],
        num_sims: int,
        simulate_from_power_rho: bool,
        **params_human: Tensor0[tf32],
):

    print("Compiling neuralcis pvalues function")

    params_human = {n: tf.repeat(p, num_sims) for n, p in params_human.items()}
    rho_ab_power = params_human.pop("rho_ab_partial_power")
    power_args = {'rho_ab_partial': rho_ab_power}

    params_net = cis._params_human_to_net(**params_human)
    params_power_net = cis._params_human_to_net(**(params_human | power_args))

    if simulate_from_power_rho:
        sim_params = params_human | power_args
    else:
        sim_params = params_human

    samples_raw = biparcorr.sampling_distribution_fn_raw(
        params_num=params_num,
        first_row_num=first_sim_num,
        num_rows=num_sims,
        seed_differently=simulate_from_power_rho,
        **sim_params,
    )
    rs = biparcorr.estimate_correlations_safe(samples_raw, params_human["n"])
    stats_human = {"rho_ab_hat": rs[:, 0],
                   "rho_bc_hat": rs[:, 1],
                   "rho_ac_hat": rs[:, 2],
                   "prop_a_hat": tf.reduce_mean(samples_raw[:, :, 0], axis=1)}
    stats_net = cis._stats_human_to_net(**stats_human)

    ps_null = cis.pnet.p(stats_net, params_net)
    ps_power = cis.pnet.p(stats_net, params_power_net)

    ps = tf.stack([ps_null, ps_power], axis=1)

    return ps


def run_neural_ps(
        method_name: str = "neural",
        num_sims_per_param_sample: int = 1_000_000,
        start_from_param_num: int = 0,
        batch_size: int = 1_000_000,
        simulate_from_power_rho: bool = False,  # for idealized power
) -> None:

    # TODO: This func should probably be factored to allow step 3 to reuse!!
    assert batch_size <= num_sims_per_param_sample

    params = biparcorr.load_params_dict()
    num_param_samples = biparcorr.get_num_param_samples(params)
    params.pop('target_power')

    assert num_sims_per_param_sample % batch_size == 0

    print("Loading trained net!")
    cis = NeuralCIs.load('../saved_model/')

    print("Compiling!")
    # Do an initial run to force a compile so that our timings are pure
    pvalues_for_batch(cis,
                      tf.constant(0, tf.int32),
                      tf.constant(0, tf.int32),
                      batch_size,
                      simulate_from_power_rho,
                      **{n: p[0] for n, p in params.items()})

    batch_size_tf = tf.constant(batch_size)
    for params_sample_num in tqdm(range(start_from_param_num,
                                        num_param_samples)):
        batch_likelihoods = []
        params_sample_num_tf = tf.constant(params_sample_num, tf.int32)
        # TODO: All this shit used to be nicely factored into biparcorr lib!!
        this_params = {n: p[params_sample_num] for n, p in params.items()}
        for batch_num in range(num_sims_per_param_sample // batch_size):
            this_ps = pvalues_for_batch(cis,
                                        params_sample_num_tf,
                                        batch_size_tf * batch_num,
                                        batch_size,
                                        simulate_from_power_rho,
                                        **this_params)
            batch_likelihoods.append(this_ps)

        filename = biparcorr.data_filename(method_name, "ps",
                                           params_sample_num)
        path = biparcorr.convert_relative_path(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        likelihoods = tf.concat(batch_likelihoods, axis=0)
        np.save(path, likelihoods.numpy())


if __name__ == "__main__":
    run_neural_ps("neural", 1_000_000)
    run_neural_ps("neural_powersim",
                  num_sims_per_param_sample=10_000,
                  batch_size=10_000,
                  simulate_from_power_rho=True)
