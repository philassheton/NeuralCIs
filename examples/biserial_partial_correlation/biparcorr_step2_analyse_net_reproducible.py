import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from neuralcis import NeuralCIs
import biparcorr_analyse_funcs as biparcorr

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd, tfb = tfp.distributions, tfp.bijectors

from tqdm import tqdm
import time

# typing
from typing import Dict
from neuralcis.common import Batch, Samples, Stats, Ys
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3
from tensor_annotations.tensorflow import float32 as tf32, int64 as ti64


# No need for N_MIN as they are sampled from the network
DO_L_BFGS = True
PARAMS_NUM_TO_CONTINUE_FROM = 0
NUM_PARAM_SAMPLES = 1_000
NUM_SIMULATIONS_PER_PARAM = 10_000_000
BATCH_SIZE = 1_000


def ps_neural_tf(
        cis: NeuralCIs,
        stats: Tensor3[tf32, Batch, Samples, Stats],
        batch_size: int,
        **params: Tensor0[tf32],
):

    params_human = {name: tf.repeat(param, (batch_size,))
                    for name, param in params.items()}

    correlations = biparcorr.estimate_correlations_safe(stats, params['n'])
    rho_ab_hat, rho_bc_hat, rho_ac_hat = tf.unstack(correlations, axis=1)
    prop_a_hat: Tensor1 = tf.reduce_sum(stats[:, :, 0], axis=1) / params['n']  # type: ignore

    params_net = cis._params_human_to_net(**params_human)
    stats_net = cis._stats_human_to_net(
        rho_ab_hat=rho_ab_hat,
        rho_bc_hat=rho_bc_hat,
        rho_ac_hat=rho_ac_hat,
        prop_a_hat=prop_a_hat,
    )

    ps = cis.pnet.p(stats_net, params_net)

    return ps


@tf.function(jit_compile=True)
def ps_for_batch(
    rho_ab_partial: Tensor0[tf32],
    rho_bc: Tensor0[tf32],
    rho_ac: Tensor0[tf32],
    prop_a: Tensor0[tf32],
    n: Tensor0[ti64],
    rho_ab_partial_power: Tensor0[tf32],
    cis: NeuralCIs,
    batch_size: int,
    generator: tf.random.Generator,
) -> Tensor2[tf32, Batch, Ys]:

    stats = biparcorr.sampling_distribution_fn_raw(
        rho_ab_partial=rho_ab_partial,
        rho_bc=rho_bc,
        rho_ac=rho_ac,
        prop_a=prop_a,
        n=n,
        batch_size=batch_size,
        generator=generator,
    )

    ps_neural_null = ps_neural_tf(cis, stats, batch_size,
                                  rho_ab_partial=rho_ab_partial,
                                  rho_bc=rho_bc,
                                  rho_ac=rho_ac,
                                  prop_a=prop_a,
                                  n=tf.cast(n, tf.float32))
    ps_neural_power = ps_neural_tf(cis, stats, batch_size,
                                   rho_ab_partial=rho_ab_partial_power,
                                   rho_bc=rho_bc,
                                   rho_ac=rho_ac,
                                   prop_a=prop_a,
                                   n=tf.cast(n, tf.float32))
    ps_neural = tf.stack([ps_neural_null, ps_neural_power], axis=1)

    return ps_neural


def rho_for_power(
        rho_null,
        n,
        signed_target_power,
        alpha: float = 0.05,
):
    normal = tfp.distributions.Normal(0., 1.)

    target_power = tf.math.abs(signed_target_power)
    alt_sign = tf.sign(signed_target_power)
    alpha = tf.constant(alpha, dtype=tf.float32)
    num_control_vars = tf.constant(1, dtype=tf.float32)

    z_alpha = normal.quantile(1.0 - alpha / 2.0)
    z_beta = normal.quantile(target_power)
    delta_z = (z_alpha + z_beta) / tf.sqrt(n - num_control_vars - 3.0)
    z_null = tf.atanh(tf.clip_by_value(rho_null, -0.999, 0.999))

    z_alt = z_null + alt_sign * delta_z
    rho_alt = tf.tanh(z_alt)

    return rho_alt


def neural_pvalues_for_one_params(
        params_human: dict[str, Tensor0[tf32]],
        param_sample_index: int,
        num_simulations: int,
        cdf_summary_length: int = 1000,
        batch_size: int = 500,
):

    generator = tf.random.Generator.from_seed(param_sample_index,
                                              tf.random.Algorithm.PHILOX)

    ps = []
    num_batches = num_simulations // batch_size
    start = time.perf_counter()
    for batch_num in tqdm(range(num_batches)):
        ps_batch = ps_for_batch(
            params_human['rho_ab_partial'],
            params_human['rho_bc'],
            params_human['rho_ac'],
            params_human['prop_a'],
            tf.cast(params_human['n'], tf.int64),
            params_human['rho_ab_partial_power'],
            cis,
            batch_size,
            generator,
        )
        ps.append(ps_batch)
    ps = tf.concat(ps, axis=0)

    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} seconds")
    print(f"Processed parameters: {params_human}")
    return biparcorr.make_cdf_summary(ps, cdf_summary_length)


def run_net_and_save_pvalue_summaries_for_params(
        save_directory: str,
        params_human: dict[str, Tensor1[tf32, Samples]],
        params_num_to_continue_from: int,
        num_simulations_per_param: int,
        cdf_summary_length: int = 1000,
        batch_size: int = 500,
) -> None:

    num_param_samples = biparcorr.get_num_param_samples(params_human)

    for params_num in range(params_num_to_continue_from, num_param_samples):
        print(params_num)
        this_params = {name: param[params_num]
                       for name, param in params_human.items()}

        summary_tensor = neural_pvalues_for_one_params(
            params_human=this_params,
            param_sample_index=params_num,
            num_simulations=num_simulations_per_param,
            cdf_summary_length=cdf_summary_length,
            batch_size=batch_size,
        )
        summary_name = biparcorr.param_run_filename(
            save_directory,
            'neur POW',
            params_num,
            this_params,
            num_simulations_per_param
        )
        np.save(summary_name, summary_tensor.numpy())


def load_or_generate_params_dict(
        param_samples_file: str,
        cis: NeuralCIs,
        num_samples_if_no_file: int,
) -> Dict[str, Tensor1[tf32, Samples]]:

    if os.path.exists(param_samples_file):
        print('Loading previous param samples!')
        params = biparcorr.load_params_dict(param_samples_file)
    else:
        print('Generating new param samples!!')
        params = cis.sample_params(NUM_PARAM_SAMPLES)
        power_targets = np.random.choice([-0.5, -0.8,
                                          0.50, 0.80], num_samples_if_no_file)
        params['target_power'] = tf.convert_to_tensor(power_targets,
                                                      dtype=tf.float32)
        params['rho_ab_partial_power'] = rho_for_power(
            params['rho_ab_partial'],
            params['n'],
            params['target_power'],
            alpha=0.05)
        biparcorr.save_params_dict(param_samples_file, params)

    return params


save_directory = 'param_runs'
param_samples_file = 'param_samples.npy'

os.makedirs(save_directory, exist_ok=True)
cis = NeuralCIs.load('saved_model', 'testing')
params = load_or_generate_params_dict(param_samples_file, cis,
                                      NUM_PARAM_SAMPLES)

run_net_and_save_pvalue_summaries_for_params(
    save_directory=save_directory,
    params_human=params,
    params_num_to_continue_from=PARAMS_NUM_TO_CONTINUE_FROM,
    num_simulations_per_param=NUM_SIMULATIONS_PER_PARAM,
    cdf_summary_length=1000,
    batch_size=BATCH_SIZE,
)
