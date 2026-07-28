import behfish_analyse_funcs as behfish
from neuralcis import NeuralCIs, comparisons_funcs

import numpy as np
import tensorflow as tf
import os

from tqdm import tqdm

# typing
from typing import Dict
from tensor_annotations.tensorflow import Tensor0, Tensor1
from tensor_annotations.tensorflow import float32 as tf32

import typing
from tensor_annotations import axes
Samples = typing.NewType("Samples", axes.Axis)


@tf.function(jit_compile=True)
def pvalues_for_batch(
        cis: NeuralCIs,
        num_sims: int,
        stats_human: Dict[str, Tensor1[tf32, Samples]],
        **params_human: Tensor0[tf32],
):

    print("Compiling neuralcis pvalues function")

    params_human = {n: tf.repeat(p, num_sims) for n, p in params_human.items()}
    mudiff_power = params_human.pop("mudiff_power")
    power_args = {'mudiff': mudiff_power}

    params_human_power = params_human | power_args

    params_net = cis._params_human_to_net(**params_human)
    params_power_net = cis._params_human_to_net(**params_human_power)

    stats_net = cis._stats_human_to_net(**stats_human)

    ps_null = cis.pnet.p(stats_net, params_net)
    ps_power = cis.pnet.p(stats_net, params_power_net)

    ps = tf.stack([ps_null, ps_power], axis=1)

    return ps


@tf.function(jit_compile=False)  # Cannot jit compile gammas
def get_random_samples(mudiff, sigma1, sigma2, n1, n2,
                       params_num, num_samples, seed_differently: bool):

    print("Compiling get_random_samples")

    return behfish.sampling_distribution_fn_stateless(
        mudiff, sigma1, sigma2, n1, n2,
        params_num=params_num,
        num_rows=num_samples,
        seed_differently=seed_differently,
    )


def run_neural_ps(
        method_name: str = "neural",
        num_sims_per_param_sample: int = 1_000_000,
        start_from_param_num: int = 0,
        batch_size: int = 1_000_000,
        simulate_from_power_mudiff: bool = False,  # for idealized power
) -> None:

    # TODO: This func should probably be factored to allow step 3 to reuse!!
    assert batch_size <= num_sims_per_param_sample

    params = behfish.load_params_dict()
    num_param_samples = behfish.get_num_param_samples(params)
    params.pop('target_power')

    assert num_sims_per_param_sample % batch_size == 0

    print("Loading trained net!")
    cis = NeuralCIs.load('../saved_model/')

    for params_sample_num in tqdm(range(start_from_param_num,
                                        num_param_samples)):

        this_params = {n: p[params_sample_num] for n, p in params.items()}

        sim_params = this_params.copy()
        mudiff_power = sim_params.pop("mudiff_power")
        if simulate_from_power_mudiff:
            sim_params['mudiff'] = mudiff_power
        stats = get_random_samples(params_num=tf.constant(params_sample_num),
                                   num_samples=num_sims_per_param_sample,
                                   seed_differently=simulate_from_power_mudiff,
                                   **sim_params)

        batch_ps = []
        for batch_num in range(num_sims_per_param_sample // batch_size):

            start = batch_num * batch_size
            end = start + batch_size
            this_stats = {n:p[start:end] for n, p in stats.items()}

            this_ps = pvalues_for_batch(cis,
                                        batch_size,
                                        this_stats,
                                        **this_params)
            batch_ps.append(this_ps)

        filename = behfish.data_filename(method_name, "ps",
                                         params_sample_num)
        path = behfish.convert_relative_path(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ps = tf.concat(batch_ps, axis=0)
        comparisons_funcs.save_pvalues_as_numpy_uint16(path, ps.numpy())


if __name__ == "__main__":
    run_neural_ps("neural", 1_000_000)
    run_neural_ps("neural_powersim",
                  num_sims_per_param_sample=10_000,
                  batch_size=10_000,
                  simulate_from_power_mudiff=True)
