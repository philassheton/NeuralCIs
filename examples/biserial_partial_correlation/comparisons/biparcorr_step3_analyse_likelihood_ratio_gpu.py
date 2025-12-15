import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from examples.biserial_partial_correlation.comparisons import \
    biparcorr_analyse_funcs as biparcorr, \
    biparcorr_likelihood_ratio_funcs as lr

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd, tfb = tfp.distributions, tfp.bijectors



# tf.config.run_functions_eagerly(True)



from tqdm import tqdm
import time

# typing
from neuralcis.common import Samples
from tensor_annotations.tensorflow import Tensor0, Tensor1
from tensor_annotations.tensorflow import float32 as tf32


DO_L_BFGS = False
NUM_SIMULATIONS_PER_PARAM = 10_000_000
BATCH_SIZE = 1_000


def profile_pvalues_for_one_params(
        params_human: dict[str, Tensor0[tf32]],
        param_sample_index: int,
        num_simulations: int,
        cdf_summary_length: int = 1000,
        batch_size: int = 500,
):

    ps = []
    num_batches = num_simulations // batch_size
    biparcorr.start_first_batch_for_param_sample(param_sample_index)
    start = time.perf_counter()
    for batch_num in tqdm(range(num_batches)):
        ps_batch = lr.ps_for_batch(
            params_human['rho_ab_partial'],
            params_human['rho_bc'],
            params_human['rho_ac'],
            params_human['prop_a'],
            tf.cast(params_human['n'], tf.int64),
            params_human['rho_ab_partial_power'],
            batch_size,
            DO_L_BFGS,
        )
        ps.append(ps_batch)
    ps = tf.concat(ps, axis=0)

    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} seconds")
    print(f"Processed parameters: {params_human}")
    return biparcorr.make_cdf_summary(ps, cdf_summary_length)


def profile_and_save_pvalue_summaries_for_params(
        save_directory: str,
        params_human: dict[str, Tensor1[tf32, Samples]],
        num_simulations_per_param: int,
        cdf_summary_length: int = 1000,
        batch_size: int = 500,
) -> None:

    num_param_samples = biparcorr.get_num_param_samples(params_human)

    for params_num in range(num_param_samples):
        this_params = {name: param[params_num]
                       for name, param in params_human.items()}

        summary_tensor = profile_pvalues_for_one_params(
            params_human=this_params,
            param_sample_index=params_num,
            num_simulations=num_simulations_per_param,
            cdf_summary_length=cdf_summary_length,
            batch_size=batch_size,
        )

        layer_order_summary = 'lr40 80 120 120_40 POW'
        if DO_L_BFGS:
            layer_order_summary += ' bfgs POW'

        summary_name = biparcorr.param_run_filename(
            save_directory,
            layer_order_summary,
            params_num,
            this_params,
            num_simulations_per_param
        )
        np.save(summary_name, summary_tensor.numpy())


if os.getcwd().endswith('biserial_partial_correlation'):
    path = ''
elif os.getcwd().lower().endswith('NeuralCIs'):
    path = 'examples/biserial_partial_correlation'
else:
    raise Exception('What directory are we in??')

save_directory = biparcorr.convert_relative_path('param_runs')
param_samples_file = biparcorr.convert_relative_path('param_samples.npy')

os.makedirs(save_directory, exist_ok=True)
params = biparcorr.load_params_dict(param_samples_file)

profile_and_save_pvalue_summaries_for_params(
    save_directory=save_directory,
    params_human=params,
    num_simulations_per_param=NUM_SIMULATIONS_PER_PARAM,
    cdf_summary_length=1000,
    batch_size=BATCH_SIZE,
)
