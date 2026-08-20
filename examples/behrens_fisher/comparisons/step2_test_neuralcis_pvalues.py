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
        mudiff_null: Tensor0[tf32],
        mudiff_alt: Tensor0[tf32],
        **other_params_human: Tensor0[tf32],
):

    print("Compiling neuralcis pvalues function")

    mudiff_null = tf.repeat(mudiff_null, num_sims)
    mudiff_alt = tf.repeat(mudiff_alt, num_sims)
    other_params_human = {n: tf.repeat(p, num_sims)
                          for n, p in other_params_human.items()}

    params_null_net = cis._params_human_to_net(mudiff=mudiff_null,
                                               **other_params_human)
    params_alt_net = cis._params_human_to_net(mudiff=mudiff_alt,
                                              **other_params_human)

    stats_net = cis._stats_human_to_net(**stats_human)

    ps_null = cis.pnet.p(stats_net, params_null_net)
    ps_alt = cis.pnet.p(stats_net, params_alt_net)

    ps = tf.stack([ps_null, ps_alt], axis=1)

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
        use_power_param_as_null: bool = False,  # for idealized power
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
        if use_power_param_as_null:
            mudiff_null = this_params.pop("mudiff_power")
            mudiff_alt = this_params.pop("mudiff")
        else:
            mudiff_null = this_params.pop("mudiff")
            mudiff_alt = this_params.pop("mudiff_power")

        stats = get_random_samples(params_num=tf.constant(params_sample_num),
                                   num_samples=num_sims_per_param_sample,
                                   seed_differently=use_power_param_as_null,
                                   mudiff=mudiff_null,
                                   **this_params)

        batch_ps = []
        for batch_num in range(num_sims_per_param_sample // batch_size):

            start = batch_num * batch_size
            end = start + batch_size
            this_stats = {n:p[start:end] for n, p in stats.items()}

            this_ps = pvalues_for_batch(cis,
                                        batch_size,
                                        this_stats,
                                        mudiff_null,
                                        mudiff_alt,
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
                  use_power_param_as_null=True)
