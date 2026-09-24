from neuralcis import NeuralCIs
from . import comparisons_funcs as comp

import tensorflow as tf

from tqdm import tqdm

# typing
from typing import List, Dict, Callable
from tensor_annotations.tensorflow import Tensor0, Tensor1
from tensor_annotations.tensorflow import float32 as tf32, int32 as ti32
from .common import Samples


def pvalues_for_whole_param(
        cis: NeuralCIs,
        sampling_distribution_fn_raw: Callable,
        stats_human_from_samples_raw: Callable,
        params_num: Tensor0[ti32],
        num_sims_per_param: int,
        **params_human: Tensor0[tf32],
) -> Tensor1[tf32, Samples]:

    print("Compiling neuralcis pvalues function")

    params_human = {n: tf.repeat(p, num_sims_per_param)
                    for n, p in params_human.items()}

    params_net = cis._params_human_to_net(**params_human)

    samples_raw = sampling_distribution_fn_raw(
        simulation_block_num=params_num,
        num_rows=num_sims_per_param,
        **params_human,
    )
    stats_human = stats_human_from_samples_raw(params_human, samples_raw)
    stats_net = cis._stats_human_to_net(**stats_human)

    ps_null = cis.pnet.p(stats_net, params_net)
    return ps_null[:, None]


@tf.function(jit_compile=True)
def summaries_for_whole_param(
        cis: NeuralCIs,
        sampling_distribution_fn_raw: Callable,
        stats_human_from_samples_raw: Callable,
        params_num: Tensor0[ti32],
        num_sims_per_param: int,
        **params_human: Tensor0[tf32],
) -> Dict[str, Tensor0[tf32]]:

    ps = pvalues_for_whole_param(cis,
                                 sampling_distribution_fn_raw,
                                 stats_human_from_samples_raw,
                                 params_num,
                                 num_sims_per_param,
                                 **params_human)

    def error_rate(ps, alpha):
        return tf.reduce_mean(tf.cast(ps < alpha, tf.float32))

    error_rate_dict = {f"error_rate_{a:.03f}_neural": error_rate(ps, a)
                       for a in (0.01, 0.05)}
    uniformity_dict = comp.compute_uniformity_measures(ps[:, 0], "neural")
    return error_rate_dict | uniformity_dict


def run_neural_ps(
        params: Dict[str, Tensor1[tf32, Samples]],
        sampling_distribution_fn_raw: Callable,
        stats_human_from_samples_raw: Callable,
        num_param_samples: int,
        num_sims_per_param_sample: int = 1_000_000,
) -> List[Dict[str, float]]:

    for p in params.values():
        assert p.ndim == 1
        assert p.shape[0] == num_param_samples

    print("Loading trained net!")
    cis = NeuralCIs.load('../saved_model/')

    # Do an initial run to force a compile so that our timings are pure
    print("Compiling!")
    summaries_for_whole_param(cis,
                              sampling_distribution_fn_raw,
                              stats_human_from_samples_raw,
                              params_num=tf.constant(0, tf.int32),
                              num_sims_per_param=num_sims_per_param_sample,
                              **{n: p[0] for n, p in params.items()})
    print("Done compiling.")

    summary_dicts = []
    for params_sample_num in tqdm(range(len(params["n"]))):
        params_sample_num_tf = tf.constant(params_sample_num, tf.int32)
        this_params = {n: p[params_sample_num] for n, p in params.items()}
        summary_dict = summaries_for_whole_param(cis,
                                                 sampling_distribution_fn_raw,
                                                 stats_human_from_samples_raw,
                                                 params_sample_num_tf,
                                                 num_sims_per_param_sample,
                                                 ** this_params)
        summary_dict = {k:v.numpy().item() for k, v in summary_dict.items()}
        summary_dicts.append(summary_dict)
    return summary_dicts
