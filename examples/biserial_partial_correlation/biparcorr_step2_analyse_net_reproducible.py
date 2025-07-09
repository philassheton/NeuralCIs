import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from neuralcis import NeuralCIs

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd, tfb = tfp.distributions, tfp.bijectors

from tqdm import tqdm
from datetime import datetime
import time

# typing
from typing import Dict
from neuralcis.common import Batch, Samples, Stats, One, Ys
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3
from tensor_annotations.tensorflow import float32 as tf32, int64 as ti64


# No need for N_MIN as they are sampled from the network
N_MAX = 100
DO_L_BFGS = True
NUM_PARAM_SAMPLES = 1_000
NUM_SIMULATIONS_PER_PARAM = 10_000_000
BATCH_SIZE = 1_000


def sampling_distribution_fn_raw(
        rho_ab_partial: Tensor0[tf32],
        rho_bc: Tensor0[tf32],
        rho_ac: Tensor0[tf32],
        prop_a: Tensor0[tf32],
        n: Tensor0[ti64],
        batch_size: int,
        generator: tf.random.Generator,
) -> Tensor3[tf32, Batch, Samples, Stats]:

    rho_ab = (rho_bc * rho_ac
              + rho_ab_partial * tf.sqrt((1. - tf.square(rho_bc)) *
                                         (1. - tf.square(rho_ac))))

    z_threshold_a = -tfp.distributions.Normal(0., 1.).quantile(prop_a)

    one = tf.ones_like(rho_ab)
    correlation_matrix = tf.stack([
        tf.stack([one, rho_ab, rho_ac], axis=0),
        tf.stack([rho_ab, one, rho_bc], axis=0),
        tf.stack([rho_ac, rho_bc, one], axis=0),
    ], axis=1)

    cholesky = tf.linalg.cholesky(correlation_matrix)

    z = generator.normal((batch_size, 3, N_MAX)) * n_mask(n)[None, None, :]
    z_correlated = tf.linalg.matmul(cholesky, z)

    a, b, c = tf.split(z_correlated, 3, axis=1)
    a = tf.cast(a > z_threshold_a, tf.float32) * n_mask(n)[None, None, :]
    samples = tf.stack([a[:, 0, :], b[:, 0, :], c[:, 0, :]], axis=2)

    return samples


# n_mask ensures we only have n z values (the rest will be zeroed)
#  -- this means we work with constant memory size.
def n_mask(n: Tensor0[ti64]) -> Tensor1[tf32, Samples]:
    return tf.cast(tf.range(N_MAX, dtype=tf.int64) < n, tf.float32)


def n_mask_safe(n: Tensor0) -> Tensor1[tf32, Samples]:
    return n_mask(tf.cast(n, tf.int64))


def estimate_correlations_safe(
        stats_tensor: Tensor3[tf32, Batch, Samples, Stats],
        n_float: Tensor0[tf32],
) -> Tensor2[tf32, Batch, 3]:

    # Returns zero correlation whenever all a values are the same.

    stats_tensor = stats_tensor * n_mask_safe(n_float)[None, :, None]          # type: ignore
    stats_mean = tf.reduce_sum(stats_tensor, axis=1, keepdims=True) / n_float
    X = (stats_tensor - stats_mean) * n_mask_safe(n_float)[None, :, None]

    X_sum_pairwise = tf.reduce_sum(X * tf.gather(X, [1, 2, 0], axis=2), axis=1)
    X_sum_sq = tf.reduce_sum(tf.square(X), axis=1)
    X_sum_sq_pairwise = X_sum_sq * tf.gather(X_sum_sq, [1, 2, 0], axis=1)

    correlations_hat = X_sum_pairwise / tf.sqrt(X_sum_sq_pairwise + 1e-10)

    return correlations_hat


def ps_neural_tf(
        cis: NeuralCIs,
        stats: Tensor3[tf32, Batch, Samples, Stats],
        batch_size: int,
        **params: Tensor0[tf32],
):

    params_human = {name: tf.repeat(param, (batch_size,))
                    for name, param in params.items()}

    correlations = estimate_correlations_safe(stats, params['n'])
    rho_ab_hat, rho_bc_hat, rho_ac_hat = tf.unstack(correlations, axis=1)
    prop_a_hat = tf.reduce_sum(stats[:, :, 0], axis=1) / params['n']

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

    stats = sampling_distribution_fn_raw(
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


def make_cdf_summary(
        ps: Tensor2[tf32, Samples, Stats],
        summary_length: int,
) -> Tensor3[tf32, One, Stats, Samples]:

    ps_sorted = tf.sort(tf.transpose(ps), axis=1)
    end_of_bucket_index = tf.linspace(0., len(ps), summary_length + 1)[1:] - 1
    end_of_bucket_index = tf.cast(end_of_bucket_index, tf.int32)
    return tf.gather(ps_sorted, end_of_bucket_index, axis=1)[None, :, :]


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
    return make_cdf_summary(ps, cdf_summary_length)


def run_net_and_save_pvalue_summaries_for_params(
        save_directory: str,
        params_human: dict[str, Tensor1[tf32, Samples]],
        num_param_samples: int,
        num_simulations_per_param: int,
        cdf_summary_length: int = 1000,
        batch_size: int = 500,
) -> None:

    for params_num in range(num_param_samples):
        this_params = {name: param[params_num]
                       for name, param in params_human.items()}

        summary_tensor = neural_pvalues_for_one_params(
            params_human=this_params,
            param_sample_index=params_num,
            num_simulations=num_simulations_per_param,
            cdf_summary_length=cdf_summary_length,
            batch_size=batch_size,
        )
        name_start = f'pars{params_num} neur POW'
        power_text = f'{int(this_params["target_power"]*100):+d}'
        summary_name = (f'{save_directory}/{name_start}'
                        f' {datetime.now().strftime("%Y%m%d %H%M%S")}'
                        f' r_ab_p {this_params["rho_ab_partial"]:.4f}'
                        f' r_ab_p{power_text}'
                        f' {this_params["rho_ab_partial_power"]:.4f}'
                        f' r_bc {this_params["rho_bc"]:.4f}'
                        f' r_ac {this_params["rho_ac"]:.4f}'
                        f' p_a {this_params["prop_a"]:.4f}'
                        f' n {int(this_params["n"]):d}'
                        f' runs {num_simulations_per_param}')
        np.save(summary_name, summary_tensor.numpy())


def load_or_generate_params_dict(
        param_samples_file: str,
        cis: NeuralCIs,
        num_samples_if_no_file: int,
) -> Dict[str, Tensor1[tf32, Samples]]:

    param_names = ['rho_ab_partial', 'rho_bc', 'rho_ac', 'prop_a', 'n',
                   'rho_ab_partial_power', 'target_power']

    if os.path.exists(param_samples_file):
        params_grid = tf.convert_to_tensor(np.load(param_samples_file),
                                           dtype=tf.float32)
        param_tensors = tf.unstack(params_grid, axis=1)
        params = {name: param for name, param in
                  zip(param_names, param_tensors)}
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
        param_tensors = [params[name] for name in param_names]
        params_grid = tf.stack(param_tensors, axis=1)
        np.save(param_samples_file, params_grid.numpy())

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
    num_param_samples=NUM_PARAM_SAMPLES,
    num_simulations_per_param=NUM_SIMULATIONS_PER_PARAM,
    cdf_summary_length=1000,
    batch_size=BATCH_SIZE,
)
