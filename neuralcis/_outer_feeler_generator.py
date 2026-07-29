from ._sampling_feeler_generator import _SamplingFeelerGenerator
from ._utils import known_params_from_params
from . import common

import tensorflow as tf
import tensorflow_probability as tfp

# typing
from typing import Optional, Callable, Tuple
from .common import Samples, Stats, Params, UnknownParams, KnownParams, Chains
from tensor_annotations.tensorflow import Tensor1, Tensor2, Tensor3
from tensor_annotations import tensorflow as ttf

tf32 = ttf.float32




class _OuterFeelerGenerator(_SamplingFeelerGenerator):
    jit_compile = False  # For some reason, getting a memory blow up with XLA
    
    def __init__(
            self,
            sample_params_inner_fn: Callable[
                [int],
                Tensor2[tf32, Samples, Params],
            ],
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],  # params
                Tensor2[tf32, Samples, Stats],     # -> ys
            ],
            preprocess_params_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor2[tf32, Samples, Params]
            ],
            inside_inner_fn: Callable[
                [Tensor2[tf32, Samples, Stats],
                 Tensor2[tf32, Samples, Params]],
                Tensor1[ttf.bool, Samples],
            ],
            params_is_valid_fn: Callable[
                [Tensor2[tf32, Samples, Params],
                 Optional[bool]],
                Tensor2[ttf.bool, Samples, Params]
            ],
            estimates_fn: Callable[
                [Tensor2[tf32, Samples, Stats],
                 Tensor2[tf32, Samples, KnownParams]],
                Tensor2[tf32, Samples, UnknownParams],
            ],
            num_unknown_param: int,
            num_known_param: int,
            profile: str,
            sample_size: int = common.SAMPLES_PER_TEST_PARAM,
            sd_known: float = common.KNOWN_PARAM_MARKOV_CHAIN_SD,
            num_chains: int = common.FEELER_NET_NUM_CHAINS,
            chain_length: int = common.FEELER_NET_MARKOV_CHAIN_LENGTH,
            peripheral_batch_size: int =
                                     common.OUTER_FEELER_PERIPHERAL_BATCH_SIZE,
            num_peripheral_batches: int =
                                     common.OUTER_FEELER_PERIPHERAL_BATCHES,
            regularize_jitter_multiply: float = 0.,
            regularize_jitter_add: float = 0.,
    ):

        _SamplingFeelerGenerator.__init__(
            self,
            sampling_distribution_fn,
            preprocess_params_fn,
            params_is_valid_fn,
            estimates_fn,
            num_unknown_param,
            num_known_param,
            profile,
            sample_size,
            sd_known,
            num_chains,
            chain_length,
            peripheral_batch_size,
            num_peripheral_batches,
            regularize_jitter_multiply,
            regularize_jitter_add,
        )

        self.sample_params_inner_fn = sample_params_inner_fn
        self.inside_inner_fn = inside_inner_fn

    def sample_starting_params(
            self, n: int,
    ) -> Tensor2[tf32, Chains, Params]:

        return self.sample_params_inner_fn(n)


    # TODO: Parts of the peripheral sampling had to be converted
    #       non-tf.functions because there was some kind of memory leak.  Look
    #       back at this again.
    def sample_statistics(
            self,
            params: Tensor2[tf32, Chains, Params],
    ) -> Tuple[
        Tensor2[tf32, Chains, Params],
        Tensor2[tf32, Chains, UnknownParams],
        Tensor3[tf32, Chains, UnknownParams, UnknownParams],
        Tensor3[tf32, Chains, UnknownParams, UnknownParams],
        Tensor1[tf32, Chains],
        Tensor1[tf32, Chains],
    ]:

        num_chains, _ = params.shape
        params_pp = self.preprocess_params_fn(params)
        params_repeated = tf.repeat(params_pp, self.sample_size, axis=0)
        known_params_repeated = known_params_from_params(
            params_repeated,
            self.num_unknown_param,
            self.num_known_param,
        )

        stats = self.sampling_distribution_fn(params_repeated)
        estimates = self.estimates_fn(stats, known_params_repeated)

        inside_inner = self.inside_inner_fn(stats, params_repeated)
        inside_inner_grouped = tf.reshape(inside_inner, (num_chains,
                                                         self.sample_size))
        hits_inner = tf.reduce_any(inside_inner_grouped, axis=1)
        hits_inner = tf.cast(hits_inner, tf.float32)

        estimates_grouped = tf.reshape(estimates, (num_chains,
                                                   self.sample_size,
                                                   self.num_estimate()))
        xbar = tf.reduce_mean(estimates_grouped, axis=1)

        if self.regularize_jitter:
            estimates_std = tf.math.reduce_std(estimates_grouped, 1,
                                               keepdims=True)
            estimates_grouped += tf.random.normal(
                estimates_grouped.shape,
                stddev=self.regularize_jitter_multiply * estimates_std
                       + self.regularize_jitter_add,
            )

        l = tfp.stats.cholesky_covariance(estimates_grouped, sample_axis=1)
        identity = tf.eye(self.num_estimate(), batch_shape=(num_chains, ))
        inv_l = tf.linalg.triangular_solve(l, identity)
        det_l = tf.reduce_prod(tf.linalg.diag_part(l), axis=1)

        return params_pp, xbar, l, inv_l, det_l, hits_inner
