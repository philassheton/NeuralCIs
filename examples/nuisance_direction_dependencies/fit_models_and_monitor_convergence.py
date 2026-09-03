import neuralcis
from neuralcis import common, comparisons_funcs, _callbacks
from neuralcis import Stat, Param, Interest
from neuralcis import Location
from neuralcis._z_net import _ZNet

import tensorflow as tf
import tensorflow_probability as tfp

import datetime
import functools


NUM_M = 32
NUM_U = 64
EPOCHS = 500
STEPS_PER_EPOCH = 100
NUM_RUNS = 250


def sampling_distribution_fn(psi, lambd):
    normal = tf.random.normal(tf.shape(psi))
    uniform = tf.random.uniform(tf.shape(psi)) - 0.5
    x2 = lambd + uniform
    x1 = psi + normal + x2
    return {"x1": x1, "x2": x2}


def interest_fn(psi, lambd):
    return psi


def estimates_fn(x1, x2):
    return {"psi": x1 - x2, "lambd": x2}


cis = neuralcis.NeuralCIs(
    sampling_distribution_fn,
    interest_fn,
    estimates_fn,

    # PUT x1, x2 and psi on the SAME SCALE for simplicity!!
    psi=Param(Location(), (-2.2, 2.2)),
    lambd=Param(Location(), (-1., 1.)),
    x1=Stat(Location()),
    x2=Stat(Location()),
    interest=Interest(Location()),
)
cis.param_sampler.fit()


###############################################################################
#  Monkey patch custom test step and fit method in
###############################################################################


def ks_uniform(u):
    u = tf.sort(u)

    n_int = tf.size(u)
    n = tf.cast(n_int, u.dtype)

    i = tf.cast(tf.range(1, n_int + 1), u.dtype)

    d_plus = tf.reduce_max(i / n - u)
    d_minus = tf.reduce_max(u - (i - 1.0) / n)

    return tf.maximum(d_plus, d_minus)


def compute_test_values_at_theta(znet_self, params):
    psi = params[0]
    lambd = params[1]
    x2 = lambd + znet_self.validation_e2
    x1 = psi + x2 + znet_self.validation_e1
    stats = cis._stats_human_to_net(x1=x1, x2=x2)
    num_samples = NUM_M * NUM_U
    params = cis._params_human_to_net(
        psi=tf.zeros((num_samples,)) + psi,
        lambd=tf.zeros((num_samples,)) + lambd,
        dummy_sample_size=tf.ones((num_samples,)),
    )
    z = znet_self.z(stats, params)
    ps = tfp.distributions.Normal(0., 1.).cdf(z)
    z_given_m = tf.reshape(z, (NUM_M, NUM_U))
    var_z_given_m = tf.math.reduce_variance(z_given_m, axis=1)

    return {
        "e__var_z_given_m": tf.math.reduce_mean(var_z_given_m),
        "e_z_given_m": tf.math.reduce_mean(z_given_m, axis=1),
        "var_z": tf.math.reduce_variance(z),
        "ks": ks_uniform(ps),
    }

def test_step(self, data):
    psis, lambdas = tf.meshgrid(tf.linspace(-2.2, 2.2, 16),
                                tf.linspace(-1., 1., 16),
                                indexing="ij")
    psis = tf.reshape(psis, [-1])
    lambdas = tf.reshape(lambdas, [-1])
    test_fn = functools.partial(compute_test_values_at_theta, self)
    test_values = tf.map_fn(
        test_fn,
        tf.stack([psis, lambdas], axis=1),
        fn_output_signature={
            "e__var_z_given_m": tf.TensorSpec(shape=(), dtype=tf.float32,
                                                       name=None),
            "e_z_given_m": tf.TensorSpec(shape=(NUM_M,), dtype=tf.float32,
                                                         name=None),
            "var_z": tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
            "ks": tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
        },
    )

    max_var_ratio = tf.reduce_max(test_values["e__var_z_given_m"]
                                  / test_values["var_z"])
    max_e__var_z_given_m = tf.reduce_max(test_values["e__var_z_given_m"])
    var__e_z_given_m = tf.math.reduce_variance(test_values["e_z_given_m"],
                                              axis=0)
    e_var__e_z_given_m = tf.reduce_max(var__e_z_given_m)
    max_var__e_z_given_m = tf.reduce_max(var__e_z_given_m)

    return {"max_var_ratio": max_var_ratio,
            "max_e__var_z_given_m": max_e__var_z_given_m,
            "max_var__e_z_given_m": max_var__e_z_given_m,
            "e_var__e_z_given_m": e_var__e_z_given_m,
            "var_z_mean": tf.reduce_mean(test_values["var_z"]),
            "var_z_var": tf.math.reduce_variance(test_values["var_z"]),
            "ks_mean": tf.reduce_mean(test_values["ks"]),
            "ks_var": tf.math.reduce_variance(test_values["ks"]),
            "ks_max": tf.reduce_max(test_values["ks"])}


def fit(
        self,
        steps_per_epoch: int = common.STEPS_PER_EPOCH,
        epochs: int = common.EPOCHS,
):

    self.get_ready_for_training()
    print(f"{datetime.datetime.now()}: Training hacked _ZNet")

    lr_scheduler = _callbacks._ReduceLROnPlateauTrackBest(
        self.simnet_weights,
        monitor=self.loss_to_watch,
        learning_rate_initial=common.LEARNING_RATE_INITIAL_ADAM,
        factor=common.LEARNING_RATE_DECAY_RATIO_ON_PLATEAU_ADAM,
        patience=common.LEARNING_RATE_PLATEAU_PATIENCE_ADAM,
        min_lr=common.LEARNING_RATE_MINIMUM_ADAM,
        absolute_loss_increase_tol=self.absolute_loss_increase_tol,
        relative_loss_increase_tol=self.relative_loss_increase_tol,
    )

    history = tf.keras.Model.fit(
        self,
        x=self.dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=2,
        callbacks=[lr_scheduler],
        validation_data=self.dataset,
        validation_steps=1,
    )

    return history


e1_probs = tf.linspace(0.5, NUM_M - 0.5, NUM_M) / NUM_M
e1_quantiles = tfp.distributions.Normal(0., 1.).quantile(e1_probs)
e1_quantiles /= tf.math.reduce_std(e1_quantiles)
e2_samples = tf.random.stateless_uniform(
    shape=(NUM_U,),
    seed=tf.constant([123, 456], dtype=tf.int32),
    minval=-0.5,
    maxval=0.5,
)

_ZNet.test_step = test_step
_ZNet.fit = fit
_ZNet.validation_e1 = tf.repeat(e1_quantiles, repeats=NUM_U)
_ZNet.validation_e2 = tf.tile(e2_samples, (NUM_M,))


for i in range(NUM_RUNS):
    znet = _ZNet(
        cis._sampling_dist_net_interface,
        cis.pnet.param_sampler.sample_params,
        cis._interest_fn_net_interface,
        cis._canonicalize_net_interface,
        num_stat=2,
        num_unknown_param=2,
        num_known_param=0,
        num_stats_remaining_after_canonicalization=2,
        profile="full",
    )

    history = znet.fit(epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename_start = f"history-{EPOCHS}x{STEPS_PER_EPOCH}-{now}"
    comparisons_funcs.save_summary_parquet(filename_start, history.history)
