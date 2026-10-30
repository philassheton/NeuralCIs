import tensorflow as tf
import numpy as np
import pandas as pd

from neuralcis import NeuralCIs

import biparcorr_analyse_funcs as biparcorr
from examples import comparisons_funcs as comp
from examples import latent_correlation_funcs as lcor

from tqdm import tqdm

# typing
from typing import List, Dict, Tuple, Sequence
from tensor_annotations.tensorflow import Tensor0, Tensor1
from tensor_annotations.tensorflow import float32 as tf32, int32 as ti32
from biparcorr_analyse_funcs import Samples


@tf.function(jit_compile=True)
def cis_for_whole_param(
        cis: NeuralCIs,
        conf_level: Tensor0[tf32],
        params_num: Tensor0[ti32],
        num_sims_per_param: int,
        **params_human: Tensor0[tf32],
) -> Tuple[Tensor1[tf32, Samples], Tensor1[tf32, Samples]]:

    print("Compiling neuralcis pvalues function")

    stats_human, params_human = biparcorr.params_human_to_stats_human_samples(
        params_num,
        num_sims_per_param,
        **params_human,
    )
    stats_net = cis._stats_human_to_net(**stats_human)
    known_params_net = cis._params_human_to_net(unknown_params=False,
                                                **params_human)

    return cis.cinet.ci(
        stats_net, known_params_net,
        target_p=tf.repeat(1. - conf_level, num_sims_per_param),
    )


def hit_rate_for_whole_param(
        cis: NeuralCIs,
        conf_level: Tensor0[tf32],
        params_num: Tensor0[ti32],
        num_sims_per_param: int,
        **params_human: Tensor0[tf32],
) -> Tensor0[tf32]:

    lower, upper = cis_for_whole_param(cis, conf_level,
                                       params_num, num_sims_per_param,
                                       **params_human)
    interest = cis.kwargs.interest_fn(
        **{n:p[None] for n, p in params_human.items()}
    )[0].numpy().item()
    hit_rate = tf.reduce_mean(tf.cast(
        (lower < interest) & (interest < upper),
        tf.float32,
    ))
    return hit_rate


def run_neural_cis(
        params: Dict[str, Tensor1[tf32, Samples]],
        conf_levels: Sequence[str] = ("95", "99"),
        num_sims_per_param_sample: int = 1_000_000,
) -> List[Dict[str, float]]:

    print("Loading trained net!")
    cis = NeuralCIs.load('../saved_model/')

    # Do an initial run to force a compile so that our timings are pure
    print("Compiling!")
    hit_rate_for_whole_param(cis,
                             conf_level=tf.constant(0.95, tf.float32),
                             params_num=tf.constant(0, tf.int32),
                             num_sims_per_param=num_sims_per_param_sample,
                             **{n: p[0] for n, p in params.items()})
    print("Done compiling.")

    summary_dicts = []
    for params_sample_num in tqdm(range(len(params["rho_ab_partial"]))):
        summary_dict = {}
        for conf_level in conf_levels:
            conf_level_tf = tf.constant(float(conf_level) / 100., tf.float32)
            params_sample_num_tf = tf.constant(params_sample_num, tf.int32)
            this_params = {n: p[params_sample_num] for n, p in params.items()}
            hit_rate = hit_rate_for_whole_param(cis,
                                                conf_level_tf,
                                                params_sample_num_tf,
                                                num_sims_per_param_sample,
                                                **this_params)
            summary_dict[conf_level] = hit_rate.numpy().item()
        summary_dicts.append(summary_dict)
    return summary_dicts


if __name__ == "__main__":
    params = lcor.sample_params()
    summary_dicts = run_neural_cis(params, ["95", "99"])
    summary_dict = {k:np.array([d[k] for d in summary_dicts])
                    for k in summary_dicts[0].keys()}
    summary_df = pd.DataFrame(summary_dict)
    params_df = pd.DataFrame(params)
    summary_df_complete = pd.concat([summary_df, params_df], axis=1)
    comp.save_summary_parquet("summary_neural_large_sample_cis",
                              summary_df_complete)
