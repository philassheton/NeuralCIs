import tensorflow as tf
import numpy as np
import pandas as pd

from neuralcis import NeuralCIs

import biparcorr_analyse_funcs as biparcorr
from examples import comparisons_funcs as comp
from examples import latent_correlation_funcs as lcor

from tqdm import tqdm

# typing
from typing import List, Dict
from tensor_annotations.tensorflow import Tensor0, Tensor1
from tensor_annotations.tensorflow import float32 as tf32, int32 as ti32
from biparcorr_analyse_funcs import Samples


def pvalues_for_whole_param(
        cis: NeuralCIs,
        params_num: Tensor0[ti32],
        num_sims_per_param: int,
        **params_human: Tensor0[tf32],
) -> Tensor1[tf32, Samples]:

    print("Compiling neuralcis pvalues function")

    stats_human, params_human = biparcorr.params_human_to_stats_human_samples(
        params_num,
        num_sims_per_param,
        **params_human,
    )
    stats_net = cis._stats_human_to_net(**stats_human)
    params_net = cis._params_human_to_net(**params_human)

    ps_null = cis.pnet.p(stats_net, params_net)
    return ps_null[:, None]


@tf.function(jit_compile=True)
def summaries_for_whole_param(
        cis: NeuralCIs,
        params_num: Tensor0[ti32],
        num_sims_per_param: int,
        **params_human: Tensor0[tf32],
) -> Dict[str, Tensor0[tf32]]:

    ps = pvalues_for_whole_param(cis, params_num, num_sims_per_param,
                                 **params_human)

    def error_rate(ps, alpha):
        return tf.reduce_mean(tf.cast(ps < alpha, tf.float32))

    error_rate_dict = {f"error_rate_{a:.03f}_neural": error_rate(ps, a)
                       for a in (0.01, 0.05)}
    uniformity_dict = comp.compute_uniformity_measures(ps[:, 0], "neural")
    return error_rate_dict | uniformity_dict


def run_neural_ps(
        params: Dict[str, Tensor1[tf32, Samples]],
        num_sims_per_param_sample: int = 1_000_000,
) -> List[Dict[str, float]]:

    print("Loading trained net!")
    cis = NeuralCIs.load('../saved_model/')

    # Do an initial run to force a compile so that our timings are pure
    print("Compiling!")
    summaries_for_whole_param(cis,
                              params_num=tf.constant(0, tf.int32),
                              num_sims_per_param=num_sims_per_param_sample,
                              **{n: p[0] for n, p in params.items()})
    print("Done compiling.")

    summary_dicts = []
    for params_sample_num in tqdm(range(len(params["rho_ab_partial"]))):
        params_sample_num_tf = tf.constant(params_sample_num, tf.int32)
        this_params = {n: p[params_sample_num] for n, p in params.items()}
        summary_dict = summaries_for_whole_param(cis,
                                                 params_sample_num_tf,
                                                 num_sims_per_param_sample,
                                                 ** this_params)
        summary_dict = {k:v.numpy().item() for k, v in summary_dict.items()}
        summary_dicts.append(summary_dict)
    return summary_dicts


if __name__ == "__main__":
    params = lcor.sample_params()
    summary_dicts = run_neural_ps(params)
    summary_dict = {k:np.array([d[k] for d in summary_dicts])
                    for k in summary_dicts[0].keys()}
    summary_df = pd.DataFrame(summary_dict)
    params_df = pd.DataFrame(params)
    summary_df = pd.concat([summary_df, params_df], axis=1)
    comp.save_summary_parquet("summary_neural_large_sample", summary_df)
