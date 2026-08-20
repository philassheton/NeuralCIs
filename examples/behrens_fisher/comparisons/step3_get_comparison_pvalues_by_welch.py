import os

import behfish_analyse_funcs as behfish
from neuralcis import comparisons_funcs

import tensorflow as tf
import tensorflow_probability as tfp

from tqdm import tqdm


@tf.function(jit_compile=True)
def welch_test(mudiff_hat, sigma1_hat, sigma2_hat,
               mudiff, n1, n2):

    print("Compiling welch test")

    s1 = sigma1_hat
    s2 = sigma2_hat

    se1 = s1 / tf.sqrt(n1)
    se2 = s2 / tf.sqrt(n2)
    se_pooled = tf.sqrt(tf.square(se1) + tf.square(se2))

    t = (mudiff_hat - mudiff) / se_pooled

    df1 = n1 - 1.
    df2 = n2 - 1.
    df_num = tf.square(tf.square(s1) / n1 + tf.square(s2) / n2)
    df_den = (tf.pow(s1, 4.) / (tf.square(n1) * df1) +
              tf.pow(s2, 4.) / (tf.square(n2) * df2))
    df = df_num / df_den

    cdf = tfp.distributions.StudentT(df, 0., 1.).cdf(t)
    p = 1. - tf.math.abs(cdf * 2. - 1)

    return p


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


def run_welch_ps(
        method_name: str,
        num_sims_per_param_sample: int,
        start_from_param_num: int = 0,
        batch_size: int = 500_000,
        use_power_param_as_null: bool = False,  # for idealized power
) -> None:

    params = behfish.load_params_dict()
    num_param_samples = behfish.get_num_param_samples(params)
    params.pop('target_power')

    assert num_sims_per_param_sample % batch_size == 0

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

            this_ps_null = welch_test(mudiff=mudiff_null,
                                      n1=this_params["n1"],
                                      n2=this_params["n2"],
                                      **this_stats)
            this_ps_alt = welch_test(mudiff=mudiff_alt,
                                     n1=this_params["n1"],
                                     n2=this_params["n2"],
                                     **this_stats)
            batch_ps.append(tf.stack([this_ps_null, this_ps_alt], axis=1))

        filename = behfish.data_filename(method_name, "ps", params_sample_num)
        path = behfish.convert_relative_path(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ps = tf.concat(batch_ps, axis=0)
        comparisons_funcs.save_pvalues_as_numpy_uint16(path, ps.numpy())


if __name__ == "__main__":
    run_welch_ps("welch", 1_000_000)
    run_welch_ps("welch_powersim",
                 num_sims_per_param_sample=10_000,
                 batch_size=10_000,
                 use_power_param_as_null=True)
