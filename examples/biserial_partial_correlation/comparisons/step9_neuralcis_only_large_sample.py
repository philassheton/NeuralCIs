import tensorflow as tf
import numpy as np
import pandas as pd

from neuralcis import NeuralCIs

import biparcorr_analyse_funcs as biparcorr
from neuralcis import comparisons_funcs as comp

from tqdm import tqdm

# typing
from typing import List, Dict
from tensor_annotations.tensorflow import Tensor0, Tensor1
from tensor_annotations.tensorflow import float32 as tf32, int32 as ti32
from biparcorr_analyse_funcs import Samples

NUM_PARAM_SAMPLES_BEFORE_WHITTLE = 20000
NUM_PARAM_SAMPLES = 10000
RHO_MAX = 0.90
PROP_A_MIN = 0.1
PROP_A_MAX = 0.9
N_MIN = 20
N_MAX = 100

PARAM_SEEDS = {
    'RHO_AB_PARTIAL': 0,
    'RHO_BC': 1,
    'RHO_AC': 2,
    'N': 3,
    'PROP_A': 4,
    'MAGNITUDE': 5,
}
SECONDARY_SEED = 42


def __sample_uniform(
        number_to_draw: int,
        param_name: str,
        minval: float,
        maxval: float,
) -> Tensor1[tf32, Samples]:
    param_seed = PARAM_SEEDS[param_name]
    seed = tf.constant([SECONDARY_SEED, param_seed], dtype=tf.int32)
    return tf.random.stateless_uniform((number_to_draw,), seed, minval, maxval)


def sample_params():
    n = tf.floor(tf.exp(
        __sample_uniform(NUM_PARAM_SAMPLES, "N",
                         np.log(N_MIN), np.log(N_MAX + 1.))
    ))
    rho_ab_partial_angle = __sample_uniform(NUM_PARAM_SAMPLES_BEFORE_WHITTLE,
                                            "RHO_AB_PARTIAL",
                                            -1., 1.)
    rho_bc_angle = __sample_uniform(NUM_PARAM_SAMPLES_BEFORE_WHITTLE,
                                    "RHO_BC",
                                    -1., 1.)
    rho_ac_angle = __sample_uniform(NUM_PARAM_SAMPLES_BEFORE_WHITTLE,
                                    "RHO_AC",
                                    -1., 1.)
    prop_a_angle = __sample_uniform(NUM_PARAM_SAMPLES_BEFORE_WHITTLE,
                                    "PROP_A",
                                    -1., 1.)
    angle = tf.stack([rho_ab_partial_angle,
                      rho_bc_angle,
                      rho_ac_angle,
                      prop_a_angle])
    angle = angle / tf.sqrt(tf.reduce_sum(angle ** 2, axis=0, keepdims=True))

    # 2 = sqrt(4) is the longest diagonal we might want to hit
    magnitude = __sample_uniform(NUM_PARAM_SAMPLES_BEFORE_WHITTLE,
                                 "MAGNITUDE",
                                 0., 2.)[None, :]
    params = angle * magnitude
    rho_ab_partial, rho_bc, rho_ac, prop_a = tf.unstack(params, axis=0)
    prop_a = prop_a / 2 + 0.5

    is_in_bounds = ((tf.math.abs(rho_ab_partial) < RHO_MAX) &
                    (tf.math.abs(rho_bc) < RHO_MAX) &
                    (tf.math.abs(rho_ac) < RHO_MAX) &
                    (PROP_A_MIN < prop_a) & (prop_a < PROP_A_MAX))

    num_surviving = tf.reduce_sum(tf.cast(is_in_bounds, tf.int16)).numpy()
    assert num_surviving >= NUM_PARAM_SAMPLES

    rho_ab_partial = tf.boolean_mask(rho_ab_partial, is_in_bounds)
    rho_bc = tf.boolean_mask(rho_bc, is_in_bounds)
    rho_ac = tf.boolean_mask(rho_ac, is_in_bounds)
    prop_a = tf.boolean_mask(prop_a, is_in_bounds)

    num_within_bounds = tf.reduce_sum(tf.cast(is_in_bounds, tf.int16))
    assert num_within_bounds >= NUM_PARAM_SAMPLES

    return {'rho_ab_partial': rho_ab_partial[0:NUM_PARAM_SAMPLES],
            'rho_bc': rho_bc[0:NUM_PARAM_SAMPLES],
            'rho_ac': rho_ac[0:NUM_PARAM_SAMPLES],
            'prop_a': prop_a[0:NUM_PARAM_SAMPLES],
            'n': n[0:NUM_PARAM_SAMPLES]}


@tf.function(jit_compile=True)
def pvalues_for_whole_param(
        cis: NeuralCIs,
        params_num: Tensor0[ti32],
        num_sims_per_param: int,
        **params_human: Tensor0[tf32],
) -> Tensor1[tf32, Samples]:

    print("Compiling neuralcis pvalues function")

    params_human = {n: tf.repeat(p, num_sims_per_param)
                    for n, p in params_human.items()}

    params_net = cis._params_human_to_net(**params_human)

    samples_raw = biparcorr.sampling_distribution_fn_raw(
        simulation_block_num=params_num,
        num_rows=num_sims_per_param,
        **params_human,
    )
    rs = biparcorr.estimate_correlations_safe(samples_raw, params_human["n"])
    prop_a_hat = biparcorr.estimate_prop_a(samples_raw, params_human["n"])
    stats_human = {"rho_ab_hat": rs[:, 0],
                   "rho_bc_hat": rs[:, 1],
                   "rho_ac_hat": rs[:, 2],
                   "prop_a_hat": prop_a_hat}
    stats_net = cis._stats_human_to_net(**stats_human)

    ps_null = cis.pnet.p(stats_net, params_net)
    return ps_null[:, None]


def run_neural_ps(
        params: Dict[str, Tensor1[tf32, Samples]],
        num_sims_per_param_sample: int = 1_000_000,
        start_from_param_num: int = 0,
) -> List[Dict[str, float]]:

    print("Loading trained net!")
    cis = NeuralCIs.load('../saved_model/')

    # Do an initial run to force a compile so that our timings are pure
    print("Compiling!")
    pvalues_for_whole_param(cis,
                            params_num=tf.constant(0, tf.int32),
                            num_sims_per_param=num_sims_per_param_sample,
                            **{n: p[0] for n, p in params.items()})
    print("Done compiling.")

    summary_dicts = []
    for params_sample_num in tqdm(range(len(params["rho_ab_partial"]))):

        params_sample_num_tf = tf.constant(params_sample_num, tf.int32)
        this_params = {n: p[params_sample_num] for n, p in params.items()}
        ps = pvalues_for_whole_param(cis,
                                     params_sample_num_tf,
                                     num_sims_per_param_sample,
                                     **this_params)

        def error_rate(ps, alpha):
            return tf.reduce_mean(tf.cast(ps < alpha, tf.float32))

        summary_dict = {f"error_rate_{a:.03f}_neural": error_rate(ps, a)
                        for a in (0.01, 0.05)}

        summary_dicts.append(summary_dict)
    return summary_dicts

if __name__ == "__main__":
    params = sample_params()
    summary_dicts = run_neural_ps(params)
    summary_dict = {k:np.array([d[k] for d in summary_dicts])
                    for k in summary_dicts[0].keys()}
    summary_df = pd.DataFrame(summary_dict)
    params_df = pd.DataFrame(params)
    summary_df = pd.concat([summary_df, params_df], axis=1)
    comp.save_summary_parquet("summary_neural_large_sample", summary_df)
