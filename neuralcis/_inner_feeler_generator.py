import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gc
from datetime import datetime

from ._data_saver import _DataSaver
from .common import FULL
from . import common

# typing
from typing import Optional, Callable, Tuple
from .common import Samples, Stats, Params, UnknownParams, KnownParams
from .common import MinAndMax, ImportanceIngredients, Chains
from tensor_annotations.tensorflow import (Tensor0, Tensor1, Tensor2,
                                           Tensor3, Tensor4)
from tensor_annotations import tensorflow as ttf

tf32 = ttf.float32
NUM_IMPORTANCE_INGREDIENTS = 2
NetInputSimulationBlob = Tuple[
    Tensor2[tf32, Samples, Params],                           # Centroid
    Tensor3[tf32, Samples, UnknownParams, UnknownParams],     # Cholesky factor
]
NetTargetBlob = Tensor2[tf32, Samples, ImportanceIngredients]


###############################################################################
#
#  TODO: This is a very very crude first cut, with loads of approximations
#        and whatnot that will need hopefully refining in a later version:
#           1) We are just defining the boundary as fixed ranges for our
#              estimates.  This will cause us some issues with the interest
#              as it will not have full information (e.g. for ANOVA, where the
#              interest makes an ellipse, but the fixed ranges make a
#              rectangular shape.  I'm still not clear about the best approach
#              between defining a range of OK parameters (in which case, I
#              think we need to then learn the volume of estimates that that
#              range of params can produce, and then further learn the range
#              of params that can throw estimates into that volume.  This
#              would allow for params to be overridden by a preference for
#              defining the interest.  HOWEVER, it does NOT allow for such an
#              easy definition of what we can and cannot put into the model
#              (since it is now defined by what parameter ranges come out --
#              though we can use the initial volume of estimates that we first
#              learned to provide a yes/no answer to whether that's OK).
#
#              The easier (but I think maybe won't quite work) approach I'm
#              using here just defines a volume of estimates that we will aim
#              to guarantee fine results if our estimates are anywhere within
#              a given range.  This provides really nice clean "that's OK to
#              use" guidelines for users and is very simple to implement (find
#              all the params that can throw estimates into that given volume)
#              but is not yet clear to me if they will get distorted when they
#              only see some of the possibilities for a given interest value.
#              That could perhaps be remedied by expanding the volume to avoid
#              that happening, but then there are still questions of how: do
#              we just crudely say that estimates passed through the interest
#              fn would be good enough as estimates of the interest here?
#              That could lead to pretty horrible results though....
#           2) Using the negative log importances and MSE makes sense in
#              in general, in particular in that the peripheral zeros will
#              not be allowed to be anything other than zero.  But this does
#              then mean that the truly "just peripheral" edges will not be
#              the nice curving trade-off between "1" and 0 that we would have
#              wanted (in the cases where some samples would hit and some
#              would miss) and that every param from which *any* sample misses
#              will be dragged to zero.  We can offset that by setting our
#              widths conservatively -- 99% of the time, sigmahat would be no
#              smaller than 82% of sigma (based on Chi-squared(99)
#              distribution) so we can set our widths to be 1/.82 times as wide
#              (see self.sd_sampling_error_adjust)
#
###############################################################################

class _InnerFeelerGenerator(_DataSaver):
    smallest_profile_found_in = FULL
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

        _DataSaver.__init__(self,
                            instance_tf_variables_to_save=("sampled_params",
                                                           "sampled_chols",
                                                           "sampled_targets",
                                                           "iteration_num"))

        self.estimates_widths = None

        if self._skip_when_profile(profile):
            return

        self.sampling_distribution_fn = sampling_distribution_fn
        self.preprocess_params_fn = preprocess_params_fn
        self.params_is_valid_fn = params_is_valid_fn
        self.estimates_fn = estimates_fn

        self.num_estimate = estimates_min_and_max.shape[0]
        self.estimates_min = estimates_min_and_max[:, 0]
        self.estimates_max = estimates_min_and_max[:, 1]
        self.estimates_widths = self.estimates_max - self.estimates_min

        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param
        self.num_param = num_unknown_param + num_known_param

        self.sample_size = sample_size
        self.sd_known = sd_known
        self.num_chains = num_chains
        self.chain_length = chain_length
        self.num_peripheral_batches = num_peripheral_batches
        self.peripheral_batch_size = peripheral_batch_size

        self.regularize_jitter = ((regularize_jitter_add > 0.)
                                  or (regularize_jitter_multiply > 0.))
        self.regularize_jitter_add = regularize_jitter_add
        self.regularize_jitter_multiply = regularize_jitter_multiply

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

        # Set up all the tf.Variables that will be used to construct the chains
        def state_variable(shape_inner, dtype=tf.float32):
            shape = [self.num_chains] + list(shape_inner)
            nans = tf.fill(shape, np.nan)
            var = tf.Variable(nans, dtype=dtype)
            return var

        num_param = self.num_param
        num_estimate = self.num_estimate

        self.params = state_variable((num_param,))
        self.mean = state_variable((num_estimate,))
        self.cov_chol = state_variable((num_estimate, num_estimate))
        self.inv_chol = state_variable((num_estimate, num_estimate))
        self.chol_det = state_variable(())
        self.importance = state_variable(())

        self.sampled_params = None
        self.sampled_chols = None
        self.sampled_targets = None

        self.iteration_num = tf.Variable(0, dtype=tf.int64)                    # tf.int32 cannot be placed on GPU

    def fit(self, *args, **kwargs):

        assert len(args) == 0
        assert len(kwargs) == 0

        if tf.greater_equal(self.iteration_num, self.chain_length):
            print(f"{datetime.now()} -- Chains Already Generated.")
            return

        print(f"{datetime.now()} -- Generating first parameter samples")
        self.initialise_for_training()

        print(f"{datetime.now()} -- Generating {self.num_chains} chains of"
              f" {self.chain_length} parameter samples")
        self.compute_chains()

        min_supported, max_supported = self.min_max_supported()

        print(f"{datetime.now()} -- Generating {self.num_peripheral_batches}"
              f" batches of {self.peripheral_batch_size} peripheral samples")
        (peripheral_params, peripheral_chols), peripheral_targets = \
            self.generate_peripheral_samples(min_supported, max_supported)

        # Compute for each sample a region around the sample that can be
        # substituted for that sample in order to smooth the surface
        self.sampled_chols, peripheral_chols = self.get_smoothing_regions(
            self.sampled_targets, self.sampled_chols,
            peripheral_targets, peripheral_chols,
            min_supported, max_supported,
        )

        print(f"{datetime.now()} -- Concatenating those")
        self.sampled_params = \
            tf.concat([self.sampled_params, peripheral_params], axis=0)
        self.sampled_chols = \
            tf.concat([self.sampled_chols, peripheral_chols], axis=0)
        self.sampled_targets = \
            tf.concat([self.sampled_targets, peripheral_targets], axis=0)

        print(f"{datetime.now()} -- Param samples generated!")

    def min_max_supported(
            self
    ) -> Tuple[Tensor1[tf32, Params],
               Tensor1[tf32, Params]]:

        supported_rows = tf.where(
            self.is_inside_support_region(self.sampled_targets)
        )[:, 0]
        params_sampled_supported = tf.gather(self.sampled_params,
                                             supported_rows, axis=0)
        mins_sampled = tf.reduce_min(params_sampled_supported, axis=0)
        maxs_sampled = tf.reduce_max(params_sampled_supported, axis=0)

        return mins_sampled, maxs_sampled

    def compute_chains(self) -> None:
        if tf.greater_equal(self.iteration_num, self.chain_length):
            print("Chains already computed!")
            return

        params, chols, targets = self.compute_chains_tf()

        num_samples = self.num_chains * self.chain_length
        self.sampled_params = tf.reshape(
            params,
            (num_samples, self.num_param),
        )
        self.sampled_chols = tf.reshape(
            chols,
            (num_samples, self.num_estimate, self.num_estimate),
        )
        self.sampled_targets = tf.reshape(
            targets,
            (num_samples, NUM_IMPORTANCE_INGREDIENTS),
        )

    @tf.function
    def compute_chains_tf(
            self
    ) -> Tuple[Tensor3[tf32, Samples, Chains, Params],
               Tensor4[tf32, Samples, Chains, UnknownParams, UnknownParams],
               Tensor3[tf32, Samples, Chains, ImportanceIngredients]]:

        params = tf.TensorArray(
            tf.float32,
            size=self.chain_length,
            element_shape=(self.num_chains,
                           self.num_param),
        )
        chols = tf.TensorArray(
            tf.float32,
            size=self.chain_length,
            element_shape=(self.num_chains,
                           self.num_estimate,
                           self.num_estimate),
        )
        targets = tf.TensorArray(
            tf.float32,
            size=self.chain_length,
            element_shape=(self.num_chains,
                           NUM_IMPORTANCE_INGREDIENTS),
        )

        for t in tf.range(self.chain_length):
            if tf.equal(t % 1000, 0):
                tf.print("step", t, "/", self.chain_length)
            params_new, chols_new, targets_new = self.sampling_iteration()
            params = params.write(t, params_new)
            chols = chols.write(t, chols_new)
            targets = targets.write(t, targets_new)

        return params.stack(), chols.stack(), targets.stack(),

    def initialise_for_training(self):
        self.iteration_num.assign(0)

        n = self.num_chains
        u_estimate = self.beta.sample((n, self.num_estimate))
        u_known_param = self.beta.sample((n, self.num_known_param))

        estimates = (u_estimate * self.estimates_widths[None, :]
                     + self.estimates_min[None, :])
        known_params = (u_known_param * (common.PARAMS_MAX - common.PARAMS_MIN)
                        + common.PARAMS_MIN)
        params = tf.concat([estimates, known_params], axis=1)

        params_pp, mean, cov_chol, inv_chol, chol_det = \
            self.sample_statistics(params)
        importance_ingredients = self.importance_ingredients(params_pp,
                                                             mean,
                                                             cov_chol,
                                                             chol_det)
        importance = self.get_importance(importance_ingredients)

        # We want to store un-preprocessed params as current state from which
        #   to step from (this way we will naturally diffuse across different
        #   discrete possibilities) but we want to store for the long term the
        #   discretized value params_pp (pp=preprocessed)_ for training our
        #   nets with.
        self.assign_iteration_results(params, mean,
                                      cov_chol, inv_chol, chol_det, importance)

    def _load_data(
            self,
            foldername: str,
            filename_start_internal: str,
            profile: str,
    ) -> None:

        if not self._skip_when_profile(profile):
            n = (self.num_chains * self.chain_length +
                 self.num_peripheral_batches * self.peripheral_batch_size)
            p = self.num_param
            e = self.num_estimate
            i = NUM_IMPORTANCE_INGREDIENTS
            print("Constructing fake CPU variables to load into")
            with tf.device("/CPU:0"):
                self.sampled_params = tf.Variable(tf.fill((n, p), np.nan),
                                                  dtype=tf.float32)
                self.sampled_chols = tf.Variable(tf.fill((n, e, e), np.nan),
                                                 dtype=tf.float32)
                self.sampled_targets = tf.Variable(tf.fill((n, i), np.nan),
                                                   dtype=tf.float32)

        super()._load_data(foldername, filename_start_internal, profile)

    def assign_iteration_results(
            self,
            params,
            mean,
            cov_chol,
            inv_chol,
            chol_det,
            importance,
    ) -> None:

        self.params.assign(params)
        self.mean.assign(mean)
        self.cov_chol.assign(cov_chol)
        self.inv_chol.assign(inv_chol)
        self.chol_det.assign(chol_det)
        self.importance.assign(importance)

    def generate_peripheral_samples(
            self,
            min_supported,
            max_supported,
    ) -> Tuple[NetInputSimulationBlob, NetTargetBlob]:

        b = self.num_peripheral_batches
        p = self.num_param
        u = self.num_unknown_param
        ni = self.peripheral_batch_size
        imp = NUM_IMPORTANCE_INGREDIENTS

        params = tf.TensorArray(tf.float32, b, element_shape=(ni, p))
        chols = tf.TensorArray(tf.float32, b, element_shape=(ni, u, u))
        targets = tf.TensorArray(tf.float32, b, element_shape=(ni, imp))

        for i in range(self.num_peripheral_batches):
            (pi, ci), ti = self.generate_peripheral_samples_batch(
                min_supported,
                max_supported,
            )
            params = params.write(i, pi)
            chols = chols.write(i, ci)
            targets = targets.write(i, ti)

        params = tf.reshape(params.stack(), (b*ni, p))
        chols = tf.reshape(chols.stack(), (b*ni, u, u))
        targets = tf.reshape(targets.stack(), (b*ni, imp))

        return (params, chols), targets

    def generate_peripheral_samples_batch(
            self,
            mins: Tensor1[tf32, Params],
            maxs: Tensor1[tf32, Params],
    ) -> Tuple[NetInputSimulationBlob, NetTargetBlob]:

        # Generate extra samples around the edges that force the
        #    probability of assigning non-zero probability at the edges
        #    down to zero.
        # TODO: Make this fit more snugly to the countours of the original
        #       sample.

        diffs = maxs - mins
        u = tf.random.uniform((self.peripheral_batch_size, self.num_param))
        sampled_params_peripheral = u * diffs[None, :] + mins[None, :]

        sampled_params_peripheral, chols_peripheral, targets_peripheral = \
            self.sample_ingredients(sampled_params_peripheral)

        sim_blob = (sampled_params_peripheral, chols_peripheral)
        return sim_blob, targets_peripheral

    def sampling_iteration(
            self,
    ) -> Tuple[Tensor2[tf32, Chains, Params],
               Tensor3[tf32, Chains, UnknownParams, UnknownParams],
               Tensor2[tf32, Chains, ImportanceIngredients]]:

        new_params = self.random_params_step(self.params, self.cov_chol)
        new_params_pp, new_mean, new_cov_chol, new_inv_chol, new_chol_det = \
            self.sample_statistics(new_params)
        prop_prob_new_given_old = self.params_proposal_pdf_proportional(
            new_params, self.params, self.inv_chol, self.chol_det,
        )
        prop_prob_old_given_new = self.params_proposal_pdf_proportional(
            self.params, new_params, new_inv_chol, new_chol_det,
        )
        new_importance_ingredients = self.importance_ingredients(new_params_pp,
                                                                 new_mean,
                                                                 new_cov_chol,
                                                                 new_chol_det)
        new_importance = self.get_importance(new_importance_ingredients)

        acceptance_prob = tf.minimum(
            1.,
            (new_importance * prop_prob_old_given_new) /
            (self.importance * prop_prob_new_given_old),
        )
        accepted = acceptance_prob > tf.random.uniform((self.num_chains,))
        accepted_2d = accepted[:, None]
        accepted_3d = accepted[:, None, None]

        params = tf.where(accepted_2d, new_params, self.params)
        mean = tf.where(accepted_2d, new_mean, self.mean)
        cov_chol = tf.where(accepted_3d, new_cov_chol, self.cov_chol)
        inv_chol = tf.where(accepted_3d, new_inv_chol, self.inv_chol)
        chol_det = tf.where(accepted, new_chol_det, self.chol_det)
        importance = tf.where(accepted, new_importance, self.importance)

        # This is not a proper importance sample
        # ...we just use the MCMC to help us decide which way to walk next
        self.assign_iteration_results(params, mean,
                                      cov_chol, inv_chol, chol_det, importance)
        self.iteration_num.assign(self.iteration_num + 1)

        return new_params_pp, new_cov_chol, new_importance_ingredients

    def random_params_step(
            self,
            params: Tensor2[tf32, Chains, Params],
            cov_chol: Tensor3[tf32, Chains, UnknownParams, UnknownParams],
    ) -> Tensor2[tf32, Chains, Params]:

        z_unknown = tf.random.normal((self.num_chains, self.num_unknown_param))
        z_known = tf.random.normal((self.num_chains, self.num_known_param))
        params_unknown = params[:, :self.num_unknown_param]
        params_known = params[:, self.num_unknown_param:]
        new_params_unknown = (
            params_unknown +
            tf.linalg.matmul(cov_chol, z_unknown[:, :, None])[:, :, 0]
        )
        new_params_known = params_known + self.sd_known * z_known
        new_params = tf.concat([new_params_unknown, new_params_known], axis=1)

        return new_params

    def sample_ingredients(
            self,
            params: Tensor2[tf32, Samples, Params],
    ) -> Tuple[
        Tensor2[tf32, Samples, Params],
        Tensor3[tf32, Samples, UnknownParams, UnknownParams],
        Tensor2[tf32, Samples, ImportanceIngredients],
    ]:

        params_preproc, mean, cov_chol, inv_chol, chol_det = \
            self.sample_statistics(params)
        importance_ingredients = self.importance_ingredients(params_preproc,
                                                             mean,
                                                             cov_chol,
                                                             chol_det)
        return params_preproc, cov_chol, importance_ingredients

    def importance_ingredients(
            self,
            params: Tensor2[tf32, Chains, Params],
            centroid: Tensor2[tf32, Chains, UnknownParams],
            cov_chol: Tensor3[tf32, Chains, UnknownParams, UnknownParams],
            chol_det: Tensor1[tf32, Chains],
    ) -> Tensor2[tf32, Chains, ImportanceIngredients]:

        eps = common.SMALLEST_LOGABLE_NUMBER

        # TODO: Since this will not anyway be the right way mathematically,
        #       just putting this in as a quick method.  If we were to stay
        #       with this approach, it should be possible to save effort
        #       by computing both together.
        overlaps = self.overlaps_estimates_box(centroid, cov_chol)

        cov_det = tf.reduce_prod(tf.linalg.diag_part(cov_chol), axis=1)
        importance_if_overlaps = tf.constant(1.) / cov_det

        # For those params outside of valid ranges, we keep the samples, so we
        #   can learn not to generate them, and zero out their importance and
        #   overlaps values, in case those are NaN values.
        params_valid_each = self.params_is_valid_fn(params)
        params_valid_all = tf.math.reduce_all(params_valid_each, axis=1)
        overlaps = tf.where(params_valid_all,
                            overlaps,
                            0.0)
        importance_if_overlaps = tf.where(params_valid_all,
                                          importance_if_overlaps,
                                          0.0)

        importance_ingredients_unlog = tf.stack([
            importance_if_overlaps,
            overlaps,
        ], axis=1)

        return tf.math.log(importance_ingredients_unlog + eps)                 # type: ignore

    def get_log_importance(
            self,
            importance_ingredients: Tensor2[tf32, Chains,
                                                  ImportanceIngredients]
    ) -> Tensor1[tf32, Chains]:

        vol = common.IMPORTANCE_INGREDIENTS_VOLUMES_INDEX
        samp = common.IMPORTANCE_INGREDIENTS_SHOULD_SAMPLE_INDEX
        return importance_ingredients[:, vol] + importance_ingredients[:, samp]

    def get_importance(
            self,
            importance_ingredients: Tensor2[tf32, Chains,
                                                  ImportanceIngredients]
    ) -> Tensor1[tf32, Chains]:

        return tf.math.exp(self.get_log_importance(importance_ingredients))

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

    def sample_statistics(
            self,
            params: Tensor2[tf32, Chains, Params],
    ) -> Tuple[
        Tensor2[tf32, Chains, Params],
        Tensor2[tf32, Chains, UnknownParams],
        Tensor3[tf32, Chains, UnknownParams, UnknownParams],
        Tensor3[tf32, Chains, UnknownParams, UnknownParams],
        Tensor1[tf32, Chains],
    ]:

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
        identity = tf.eye(self.num_estimate, batch_shape=(num_chains, ))
        inv_l = tf.linalg.triangular_solve(l, identity)
        det_l = tf.reduce_prod(tf.linalg.diag_part(l), axis=1)

        return params_pp, xbar, l, inv_l, det_l

    def draw_estimates_grouped(
            self,
            params_preprocessed: Tensor2[tf32, Chains, Params],
            num_chains: int,
    ) -> Tensor3[tf32, Chains, Samples, UnknownParams]:

        params_repeated = tf.repeat(params_preprocessed,
                                    self.sample_size,
                                    axis=0)
        known_params_repeated = params_repeated[:, -self.num_known_param:]
        stats = self.sampling_distribution_fn(params_repeated)
        estimates = self.estimates_fn(stats, known_params_repeated)
        estimates_grouped = tf.reshape(estimates, (num_chains,
                                                   self.sample_size,
                                                   self.num_estimate))

        return estimates_grouped

    def covariance_cholesky_computation(
            self,
            estimates: Tensor3[tf32, Chains, Samples, UnknownParams],
            estimates_mean: Tensor2[tf32, Chains, UnknownParams],
    ) -> Tensor3[tf32, Chains, UnknownParams, UnknownParams]:

        # TODO: Check if tfp.stats.cholesky_covariance produces stable enough
        #       output consistently to remove this function.  Currently unused
        #       but not deleting as might be needed in future.

        estimates_centred = estimates - estimates_mean[:, None, :]
        c, n, p = estimates_centred.shape
        q, r = tf.linalg.qr(estimates_centred)
        r = r * tf.sign(tf.linalg.diag_part(r))[:, :, None]
        l = tf.transpose(r, perm=(0, 2, 1)) / tf.math.sqrt(n - 1.)

        return l

    def params_proposal_pdf_proportional(
            self,
            x: Tensor2[tf32, Chains, Params],
            mu: Tensor2[tf32, Chains, Params],
            sigma_chol_inv: Tensor3[tf32, Chains, UnknownParams,
                                                  UnknownParams],
            sigma_chol_det: Tensor1[tf32, Chains],
    ) -> Tensor1[tf32, Chains]:

        d = x - mu
        d_unknown = d[:, :self.num_unknown_param]
        d_known = d[:, self.num_unknown_param:]

        z_unknown = tf.linalg.matmul(sigma_chol_inv,
                                     d_unknown[:, :, None])[:, :, 0]
        z_known = d_known / self.sd_known
        z = tf.concat([z_unknown, z_known], axis=1)
        z_norm_sq = tf.reduce_sum(tf.square(z), axis=1)

        det_unknown = sigma_chol_det
        det_known = tf.math.pow(self.sd_known, self.num_known_param)

        # NB: det is det of Cholesky factor, so no need for the usual sqrt
        return tf.math.exp(-0.5 * z_norm_sq) / (det_unknown * det_known)

    def is_inside_support_region(
            self,
            targets: Tensor2[tf32, Samples, ImportanceIngredients],
    ) -> Tensor1[ttf.bool, Samples]:

        is_inside_index = common.IMPORTANCE_INGREDIENTS_SHOULD_SAMPLE_INDEX
        return targets[:, is_inside_index] >= 0.                               # type: ignore

    def get_chol_det_from_targets(
            self,
            targets: Tensor2[tf32, Samples, ImportanceIngredients],
    ) -> Tensor1[tf32, Samples]:

        return 1. / tf.math.exp(targets[:, 0])

    def get_smoothing_regions(
            self,
            targets: Tensor2[tf32, Samples, ImportanceIngredients],
            chols: Tensor3[tf32, Samples, UnknownParams, UnknownParams],
            targets_peripheral: Tensor2[tf32, Samples, ImportanceIngredients],
            chols_peripheral: Tensor3[tf32, Samples, UnknownParams,
                                                     UnknownParams],
            mins: Tensor1[tf32, Params],
            maxs: Tensor1[tf32, Params],
    ) -> Tuple[
        Tensor3[tf32, Samples, UnknownParams, UnknownParams],
        Tensor3[tf32, Samples, UnknownParams, UnknownParams],
    ]:

        peripherals_inside = self.is_inside_support_region(targets_peripheral)
        peripherals_inside_float = tf.cast(peripherals_inside, tf.float32)
        prop_peripherals_inside = tf.reduce_mean(peripherals_inside_float)
        bounding_volume = tf.reduce_prod(maxs - mins)
        support_volume = prop_peripherals_inside * bounding_volume

        sample_is_inside = self.is_inside_support_region(targets)
        targets_inside = tf.boolean_mask(targets, sample_is_inside, axis=0)
        volumes_inside = self.get_chol_det_from_targets(targets_inside)
        total_volumes_inside = tf.reduce_sum(volumes_inside)

        # Our goal is to make the determinants of all the Cholesky factors add
        # up to the same as the support_volume, so that we will have just a
        # little overlap between each datapoint.
        chol_scale_factor = tf.math.pow(support_volume / total_volumes_inside,
                                        1. / self.num_param)
        print(f"Cholesky scale factor: {chol_scale_factor}.  If this is "
              f"above 0.5, you might want to think about increasing the "
              f"number of chains, or the chain length.")

        # But we want only to apply that to those datapoints that are INSIDE
        # the support region, since otherwise, we might eat away at the
        # support region by expanding datapoints that are outside.
        chol_scalings = tf.where(sample_is_inside, chol_scale_factor, 0.)
        chol_scalings_peripheral = tf.where(peripherals_inside,
                                            chol_scale_factor, 0.)

        chols_scaled = chols * chol_scalings[:, None, None]
        chols_scaled_peripheral = (chols_peripheral *
                                   chol_scalings_peripheral[:, None, None])

        return chols_scaled, chols_scaled_peripheral

    def release_gpu_memory(
            self,
    ) -> None:

        with tf.device("/CPU:0"):
            params_cpu = tf.identity(self.sampled_params)
            chols_cpu = tf.identity(self.sampled_chols)
            targets_cpu = tf.identity(self.sampled_targets)

        del self.sampled_params
        del self.sampled_chols
        del self.sampled_targets

        gc.collect()
        tf.keras.backend.clear_session()

        self.sampled_params = params_cpu
        self.sampled_chols = chols_cpu
        self.sampled_targets = targets_cpu
