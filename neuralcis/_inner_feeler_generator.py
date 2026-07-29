from ._sampling_feeler_generator import _SamplingFeelerGenerator
from ._utils import known_params_from_params, concat_unknown_and_known_params
from . import common

import tensorflow as tf
import tensorflow_probability as tfp

# typing
from typing import Optional, Callable, Tuple
from .common import Samples, Stats, Params, UnknownParams, KnownParams
from .common import MinAndMax, Chains
from tensor_annotations.tensorflow import Tensor1, Tensor2, Tensor3
from tensor_annotations import tensorflow as ttf

tf32 = ttf.float32


class _InnerFeelerGenerator(_SamplingFeelerGenerator):
    def __init__(
            self,
            estimates_min_and_max: Tensor2[tf32, UnknownParams, MinAndMax],
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],  # params
                Tensor2[tf32, Samples, Stats],  # -> ys
            ],
            preprocess_params_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor2[tf32, Samples, Params]
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
                                       common.FEELER_NET_PERIPHERAL_BATCH_SIZE,
            num_peripheral_batches: int = common.FEELER_NET_PERIPHERAL_BATCHES,
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

        self.estimates_widths = None

        self.estimates_min = estimates_min_and_max[:, 0]
        self.estimates_max = estimates_min_and_max[:, 1]
        self.estimates_widths = self.estimates_max - self.estimates_min

        # Choosing alpha (concentration1) and beta (concentration2) makes
        # our beta distribution here symmetric.  A value of 1. for both of
        # these params will give us a uniform distribution; a value below 1.
        # allows us to emphasise points at the edge of our distribution.
        # https://eurekastatistics.com/beta-distribution-pdf-grapher/
        beta_params = common.FEELER_GENERATOR_BETA_DISTRIBUTION_BETA_AND_ALPHA
        self.beta = tfp.distributions.Beta(concentration1=beta_params,
                                           concentration0=beta_params)

        # See note 2 at the top of this script.  This computes an adjustment
        # of 1 / .82 for a self.sample_size of 100
        self.sd_sampling_error_adjust = 1. / tf.sqrt(
            tfp.distributions.Chi2(sample_size - 1).quantile(.005)
            /
            (sample_size - 1)
        )

    def sample_starting_params(
            self, n: int,
    ) -> Tensor2[tf32, Chains, Params]:

        u_estimate = self.beta.sample((n, self.num_estimate()))
        u_known_param = self.beta.sample((n, self.num_known_param))

        estimates = (u_estimate * self.estimates_widths[None, :]
                     + self.estimates_min[None, :])
        known_params = (u_known_param * (common.PARAMS_MAX - common.PARAMS_MIN)
                        + common.PARAMS_MIN)
        params = concat_unknown_and_known_params(estimates, known_params)

        return params

    def sample_statistics(
            self,
            params: Tensor2[tf32, Chains, Params],
    ) -> Tuple[Tensor2[tf32, Chains, Params],                                  # Params (preprocessed)
               Tensor2[tf32, Chains, UnknownParams],                           # Mean
               Tensor3[tf32, Chains, UnknownParams, UnknownParams],            # L = Cholesky(Covariance matrix)
               Tensor3[tf32, Chains, UnknownParams, UnknownParams],            # L^-1
               Tensor1[tf32, Chains],                                          # |L|
               Tensor1[tf32, Chains]]:                                         # Sample blob overlaps interest region

        num_chains, _ = params.shape
        params_pp = self.preprocess_params_fn(params)
        estimates_grouped = self.draw_estimates_grouped(params_pp, num_chains)
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
        tf.debugging.check_numerics(l, "Cholesky problem!!")
        identity = tf.eye(self.num_estimate(), batch_shape=(num_chains, ))
        inv_l = tf.linalg.triangular_solve(l, identity)
        det_l = tf.reduce_prod(tf.linalg.diag_part(l), axis=1)

        overlaps = self.overlaps_estimates_box(xbar, l)

        return params_pp, xbar, l, inv_l, det_l, overlaps

    def draw_estimates_grouped(
            self,
            params_preprocessed: Tensor2[tf32, Chains, Params],
            num_chains: int,
    ) -> Tensor3[tf32, Chains, Samples, UnknownParams]:

        params_repeated = tf.repeat(params_preprocessed,
                                    self.sample_size,
                                    axis=0)
        known_params_repeated = known_params_from_params(
            params_repeated,
            self.num_unknown_param,
            self.num_known_param,
        )
        stats = self.sampling_distribution_fn(params_repeated)
        estimates = self.estimates_fn(stats, known_params_repeated)
        estimates_grouped = tf.reshape(estimates, (num_chains,
                                                   self.sample_size,
                                                   self.num_estimate()))
        return estimates_grouped

    def overlaps_estimates_box(
            self,
            centroid: Tensor2[tf32, Chains, UnknownParams],
            cov_chol: Tensor3[tf32, Chains, UnknownParams, UnknownParams],
    ) -> Tensor1[tf32, Chains]:

        # TODO: Quick substitution for now.  Instead of testing whether the
        #       sample intersects with the estimates box using the Cholesky
        #       factor, instead we just compare bounding boxes.  This will be
        #       fast and quite alright for early examples.  But we will
        #       probably need something more sophisticated as (i)
        #       dimensionality grows and (ii) as we come on to more correlated
        #       estimates.  (If we do not do this better, we will end up in
        #       some cases spending *most* of our time sampling from extreme
        #       cases that do not even produce samples within our region of
        #       interest).

        # Very crude Bonferroni adjusted bounding box for now.  Should be
        #   fine for small number of parameters.
        # TODO: Again, we need to look more carefully at this.  How do we make
        #   sure we are sampling enough but not too much, to make sure we have
        #   sufficient info?
        tails_probability = 1. - common.SAMPLE_PARAM_IF_SAMPLE_PERCENTILE / 100
        tails_probability_bonferroni = tails_probability / self.num_param
        quantile = 1 - tails_probability_bonferroni / 2.
        cutoff = tfp.distributions.Normal(0., 1.).quantile(quantile)
        cutoff = cutoff * self.sd_sampling_error_adjust                        # See comment number 2. at top of page
        sds = tf.sqrt(tf.reduce_sum(tf.square(cov_chol), axis=2))
        bounding_box_lower = centroid - cutoff * sds
        bounding_box_upper = centroid + cutoff * sds

        param_is_above_low = bounding_box_lower <= self.estimates_max[None, :]
        param_is_below_upp = bounding_box_upper >= self.estimates_min[None, :]
        params_are_above_lower = tf.math.reduce_all(param_is_above_low, axis=1)
        params_are_below_upper = tf.math.reduce_all(param_is_below_upp, axis=1)
        overlaps = params_are_above_lower & params_are_below_upper

        return tf.cast(overlaps, tf.float32)
