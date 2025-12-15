# The goal here is to generate a grid of params to test our methods against.
# In order to make sure we have nice clean sampling distributions, we will
# avoid boundary cases that can generate degenerate samples (e.g. all values
# zero or one in the binary variable).

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from examples.biserial_partial_correlation.comparisons import \
    biparcorr_analyse_funcs as biparcorr

from tensor_annotations.tensorflow import Tensor1
from tensor_annotations.tensorflow import float32 as tf32
from examples.biserial_partial_correlation.comparisons.biparcorr_analyse_funcs import Samples


NUM_PARAM_SAMPLES = 1000
TARGET_RATE = 5e-5
R_MAX = 0.99
TARGET_POWER = 0.80
ALPHA = 0.05

PARAM_SEEDS = {
    'RHO_AB_PARTIAL': 0,
    'RHO_BC': 1,
    'RHO_AC': 2,
    'N': 3,
    'PROP_A': 4,
    'UP_DOWN': 5,
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


def __prop_a_min_max(
        n: Tensor1[tf32, Samples],
        target_rate: float = TARGET_RATE,
):
    # target_rate ** 1/n, but keeping types carefully checked.
    prop_a_max = tf.pow(tf.constant(target_rate, dtype=n.dtype),
                        tf.math.reciprocal(n))
    prop_a_min = 1. - prop_a_max
    return prop_a_min, prop_a_max


def __pointbiserial_to_biserial_approx_conversion_ratio(
        prop_a: Tensor1[tf32, Samples],
) -> tf.Tensor:
    normal = tfp.distributions.Normal(0., 1.)
    z_p = normal.quantile(prop_a)  # p = P(A=1)
    phi_z = tf.exp(-0.5 * tf.square(z_p)) / tf.sqrt(2. * np.pi)
    conversion_ratio = tf.sqrt(prop_a * (1. - prop_a)) / phi_z
    return conversion_ratio


def __rho_min_max(
        n: Tensor1[tf32, Samples],
        partial: bool = False,
        biserial: bool = False,
        prop_a: Tensor1[tf32, Samples] | None = None,  # Only needed if biserial
        r_max: float = R_MAX,
        target_rate: float = TARGET_RATE,
):
    if partial:
        n = n - 1.

    if biserial:
        if prop_a is None:
            raise ValueError("prop_a required when biserial=True")

        # This ratio is not strictly correct for the partial correlation,
        #   but seems to work well enough in practice.
        pointbiseral_to_biserial = \
            __pointbiserial_to_biserial_approx_conversion_ratio(prop_a)
        r_max = r_max / pointbiseral_to_biserial

    # Because the Fisher r-to-z tends to not quite reach far enough:
    conservative_target_rate = target_rate / 2.
    std_norm = tfp.distributions.Normal(0., 1.)
    z_alpha = std_norm.quantile(1. - conservative_target_rate)
    rho_max = tf.tanh(tf.atanh(r_max) - z_alpha / tf.sqrt(n - 3.))

    if biserial:
        rho_max = rho_max * pointbiseral_to_biserial

    rho_max = tf.maximum(rho_max, 0.)
    return -rho_max, rho_max


def __rho_ab_partial_power_from_alt(
        num_param: int,
        n: Tensor1[tf32, Samples],
        prop_a: Tensor1[tf32, Samples],
        rho_ab_partial: Tensor1[tf32, Samples],  # rho_ab from which we simulate
        rho_ab_min: Tensor1[tf32, Samples],
        rho_ab_max: Tensor1[tf32, Samples],
        alpha: float,
        target_power: float,
) -> Tensor1[tf32, Samples]:

    pointbiserial_to_biserial = \
        __pointbiserial_to_biserial_approx_conversion_ratio(prop_a)
    sd_fisher = 1. / tf.sqrt(n - 1. - 3.)
    normal = tfp.distributions.Normal(0., 1.)

    # Here z will refer to a statistic with approx SD = 1 (not a Fisher z)
    def rho_biserial_to_z(rho_biserial):
        rho_pointbiserial = rho_biserial /  pointbiserial_to_biserial
        return tf.atanh(rho_pointbiserial) / sd_fisher

    def z_to_rho_biserial(z):
        rho_pointbiserial = tf.tanh(z * sd_fisher)
        return rho_pointbiserial * pointbiserial_to_biserial

    z_true = rho_biserial_to_z(rho_ab_partial)

    z_alpha = normal.quantile(1. - alpha / 2.)  # 1.96 for alpha=0.05
    z_beta = normal.quantile(target_power)  # 0.84 for target_power=0.80

    z_null_up = z_true + z_alpha + z_beta  # shift z_true enough to overcome...
    z_null_down = z_true - z_alpha - z_beta  # ...both test and required power

    z_min = rho_biserial_to_z(rho_ab_min)
    z_max = rho_biserial_to_z(rho_ab_max)

    overshoot_up = tf.keras.activations.relu(z_null_up - z_max)
    overshoot_down = tf.keras.activations.relu(z_min - z_null_down)
    both_inside = (z_null_up <= z_max) & (z_null_down >= z_min)
    tie = overshoot_up == overshoot_down

    randomise_up = __sample_uniform(num_param, 'UP_DOWN', 0., 1.) > 0.5

    choose_up = tf.where(both_inside | tie,
                         randomise_up,
                         overshoot_up < overshoot_down)

    z_null = tf.where(choose_up,
                      tf.clip_by_value(z_null_up, z_min, z_max),
                      tf.clip_by_value(z_null_down, z_min, z_max))

    return z_to_rho_biserial(z_null)


def __partial_biserial_power(
        n: Tensor1[tf32, Samples],
        prop_a: Tensor1[tf32, Samples],
        rho_ab_partial_alt: Tensor1[tf32, Samples],  # to be simulated from
        rho_ab_partial_null: Tensor1[tf32, Samples],  # to be tested against
        alpha: float = ALPHA,
) -> Tensor1[tf32, Samples]:

    normal = tfp.distributions.Normal(0., 1.)

    pointbiserial_to_biserial = \
        __pointbiserial_to_biserial_approx_conversion_ratio(prop_a)

    rho1_pb = rho_ab_partial_alt / pointbiserial_to_biserial
    rho0_pb = rho_ab_partial_null / pointbiserial_to_biserial

    z1 = tf.atanh(rho1_pb) * tf.sqrt(n - 1. - 3.)
    z0 = tf.atanh(rho0_pb) * tf.sqrt(n - 1. - 3.)

    zcrit = normal.quantile(1. - alpha / 2.)  # 1.96 for alpha=0.05

    lower_reject = z0 - zcrit
    upper_reject = z0 + zcrit

    probability_lower_rejected = normal.cdf(lower_reject - z1)
    probability_upper_rejected = 1. - normal.cdf(upper_reject - z1)
    power = probability_lower_rejected + probability_upper_rejected

    return power


def sample_n_params(
        num_param: int,
) -> dict[str, Tensor1[tf32, Samples]]:

    # Since low n cases are the most restricted (in terms of allowed
    # correlations and prop_a values), a uniform sampling regime is used for
    # n rather than the more obvious log-uniform sampling regime.
    n = tf.floor(__sample_uniform(num_param, "N", 20., 101.))

    prop_a_min, prop_a_max = __prop_a_min_max(n)
    prop_a = __sample_uniform(num_param, "PROP_A", prop_a_min, prop_a_max)

    rho_ab_min, rho_ab_max = __rho_min_max(n, biserial=True, prop_a=prop_a,
                                           partial=True)
    rho_ac_min, rho_ac_max = __rho_min_max(n, biserial=True, prop_a=prop_a)
    rho_bc_min, rho_bc_max = __rho_min_max(n)

    rho_ab_partial = __sample_uniform(num_param, "RHO_AB_PARTIAL",
                                      rho_ab_min, rho_ab_max)
    rho_bc = __sample_uniform(num_param, "RHO_BC", rho_bc_min, rho_bc_max)
    rho_ac = __sample_uniform(num_param, "RHO_AC", rho_ac_min, rho_ac_max)

    rho_ab_partial_power = __rho_ab_partial_power_from_alt(
        num_param,
        n=n,
        prop_a=prop_a,
        rho_ab_partial=rho_ab_partial,
        rho_ab_min=rho_ab_min,
        rho_ab_max=rho_ab_max,
        alpha=ALPHA,
        target_power=TARGET_POWER,
    )
    rho_ab_partial_power = tf.clip_by_value(rho_ab_partial_power,
                                            rho_ab_min,
                                            rho_ab_max)

    target_power = __partial_biserial_power(
    n=n,
    prop_a=prop_a,
    rho_ab_partial_alt=rho_ab_partial,          # true parameter
    rho_ab_partial_null=rho_ab_partial_power,   # null in the test
    alpha=0.05,
)

    return {
        "rho_ab_partial": rho_ab_partial,
        "rho_ab_partial_power": rho_ab_partial_power,
        "target_power": target_power,
        "rho_bc": rho_bc,
        "rho_ac": rho_ac,
        "prop_a": prop_a,
        "n": n,
    }

params = sample_n_params(NUM_PARAM_SAMPLES)
biparcorr.save_params_dict(params)
