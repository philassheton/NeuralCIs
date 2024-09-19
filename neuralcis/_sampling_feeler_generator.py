import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from datetime import datetime

from neuralcis._data_saver import _DataSaver
from neuralcis import common

# typing
from typing import Callable, Tuple
from neuralcis.common import Samples, Estimates, Params, UnknownParams
from neuralcis.common import MinAndMax, ImportanceIngredients, Chains
from tensor_annotations.tensorflow import Tensor1, Tensor2, Tensor3
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
#              estimates.  This will cause us some issues with the contrast
#              as it will not have full information (e.g. for ANOVA, where the
#              contrast makes an ellipse, but the fixed ranges make a
#              rectangular shape.  I'm still not clear about the best approach
#              between defining a range of OK parameters (in which case, I
#              think we need to then learn the volume of estimates that that
#              range of params can produce, and then further learn the range
#              of params that can throw estimates into that volume.  This
#              would allow for params to be overridden by a preference for
#              defining the contrast.  HOWEVER, it does NOT allow for such an
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
#              only see some of the possibilities for a given contrast value.
#              That could perhaps be remedied by expanding the volume to avoid
#              that happening, but then there are still questions of how: do
#              we just crudely say that estimates passed through the contrast
#              fn would be good enough as estimates of the contrast here?
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

class _SamplingFeelerGenerator(_DataSaver, tf.keras.Model):
    def __init__(
            self,
            estimates_min_and_max: Tensor2[tf32, Estimates, MinAndMax],
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],  # params
                Tensor2[tf32, Samples, Estimates],  # -> ys
            ],
            num_unknown_param: int,
            num_known_param: int,
            sample_size: int = common.SAMPLES_PER_TEST_PARAM,
            sd_known: float = common.KNOWN_PARAM_MARKOV_CHAIN_SD,
            num_chains: int = common.FEELER_NET_NUM_CHAINS,
            chain_length: int = common.FEELER_NET_MARKOV_CHAIN_LENGTH,
            peripheral_batch_size: int =
                                       common.FEELER_NET_PERIPHERAL_BATCH_SIZE,
            num_peripheral_batches: int = common.FEELER_NET_PERIPHERAL_BATCHES,
    ):

        tf.keras.Model.__init__(self)

        self.sampling_distribution_fn = sampling_distribution_fn
        self.num_estimate = estimates_min_and_max.shape[0]
        self.estimates_min = estimates_min_and_max[:, 0]
        self.estimates_max = estimates_min_and_max[:, 1]
        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param
        self.num_param = num_unknown_param + num_known_param

        self.sample_size = sample_size
        self.sd_known = sd_known
        self.num_chains = num_chains
        self.chain_length = chain_length
        self.num_peripheral_batches = num_peripheral_batches
        self.peripheral_batch_size = peripheral_batch_size

        # We will draw our very first param sample from the estimates box,
        #   since we are for starters assuming that the estimates are indeed
        #   estimates of the params, so that's probably a good place to start.
        self.first_params_min = tf.concat([
            self.estimates_min,
            tf.repeat(common.PARAMS_MIN, num_known_param),
        ], axis=0)
        self.first_params_widths = tf.concat([
            self.estimates_max - self.estimates_min,
            tf.repeat(common.PARAMS_MAX - common.PARAMS_MIN, num_known_param),
        ], axis=0)

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

        def samples_variable(shape_inner, dtype=tf.float32):
            # Important to have chain length as first index (if a little
            #   "wrong"-sounding, because we will want to index in by that.
            shape = [self.chain_length, self.num_chains] + list(shape_inner)
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

        self.sampled_params = samples_variable((num_param,))
        self.sampled_targets = samples_variable((NUM_IMPORTANCE_INGREDIENTS,))
        self.sampled_chols = samples_variable((num_estimate, num_estimate))

        self.peripheral_params = []
        self.peripheral_targets = []
        self.peripheral_chols = []

        self.iteration_num = tf.Variable(0, dtype=tf.int64)                    # tf.int32 cannot be placed on GPU

        _DataSaver.__init__(self,
                            instance_tf_variables_to_save=("sampled_params",
                                                           "sampled_targets",
                                                           "sampled_chols",
                                                           "iteration_num"))

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

        mins_valid, maxs_valid = self.mins_and_maxs_valid()

        print(f"{datetime.now()} -- Generating {self.num_peripheral_batches}"
              f" batches of {self.peripheral_batch_size} peripheral samples")
        (peripheral_params, peripheral_chols), peripheral_targets = \
            self.generate_peripheral_samples(mins_valid, maxs_valid)

        # Compute for each sample a region around the sample that can be
        # substituted for that sample in order to smooth the surface
        self.sampled_chols, peripheral_chols = self.get_smoothing_regions(
            self.sampled_targets, self.sampled_chols,
            peripheral_targets, peripheral_chols,
            mins_valid, maxs_valid,
        )

        print(f"{datetime.now()} -- Concatenating those")
        self.sampled_params = \
            tf.concat([self.sampled_params, peripheral_params], axis=0)
        self.sampled_chols = \
            tf.concat([self.sampled_chols, peripheral_chols], axis=0)
        self.sampled_targets = \
            tf.concat([self.sampled_targets, peripheral_targets], axis=0)

        self.min_params_valid = mins_valid
        self.max_params_simulated = maxs_valid

        print(f"{datetime.now()} -- Param samples generated!")

    def mins_and_maxs_valid(
            self
    ) -> Tuple[Tensor1[tf32, Params],
               Tensor1[tf32, Params]]:

        valid_rows = tf.where(
            self.is_inside_support_region(self.sampled_targets)
        )[:, 0]
        params_sampled_valid = tf.gather(self.sampled_params,
                                         valid_rows, axis=0)
        mins_sampled = tf.reduce_min(params_sampled_valid, axis=0)
        maxs_sampled = tf.reduce_max(params_sampled_valid, axis=0)

        return mins_sampled, maxs_sampled

    @staticmethod
    def dummy_training_stuff() -> Tuple[tf.keras.optimizers.Optimizer,
                                        tf.data.Dataset]:

        # TODO: Find a way to use all the nice Keras bits, without needing this
        dummy_optimizer = tf.keras.optimizers.SGD()
        dummy_dataset = tf.data.Dataset.from_tensor_slices((
            tf.zeros((1, 1)),
            tf.zeros((1, 1)),
        )).repeat()
        return dummy_optimizer, dummy_dataset

    @tf.function
    def train_step(self, _):
        self.sampling_iteration()
        return {}

    def compute_chains(self) -> None:
        if tf.greater_equal(self.iteration_num, self.chain_length):
            print("Chains already computed!")
            return

        dummy_optimizer, dummy_dataset = self.dummy_training_stuff()
        super().compile(optimizer=dummy_optimizer, loss=None)
        super().fit(x=dummy_dataset,
                    epochs=1,
                    steps_per_epoch=self.chain_length - 1)

        num_samples = self.num_chains * self.chain_length
        self.sampled_params = tf.reshape(
            self.sampled_params,
            (num_samples, self.num_param),
        )
        self.sampled_chols = tf.reshape(
            self.sampled_chols,
            (num_samples, self.num_estimate, self.num_estimate),
        )
        self.sampled_targets = tf.reshape(
            self.sampled_targets,
            (num_samples, NUM_IMPORTANCE_INGREDIENTS),
        )

    def initialise_for_training(self):
        self.iteration_num.assign(0)

        u = tf.random.uniform((self.num_chains, self.num_param))
        params = (u * self.first_params_widths[None, :] +
                  self.first_params_min[None, :])
        mean, cov_chol, inv_chol, chol_det = \
            self.sample_statistics(params, self.num_chains)
        importance_ingredients = self.importance_ingredients(params,
                                                             mean,
                                                             cov_chol,
                                                             chol_det)
        importance = self.get_importance(importance_ingredients)

        self.assign_iteration_results(params, mean,
                                      cov_chol, inv_chol, chol_det,
                                      importance,
                                      params, importance_ingredients, cov_chol)

    def load(self, *args, **kwargs) -> None:
        n = (self.num_chains * self.chain_length +
             self.num_peripheral_batches * self.peripheral_batch_size)
        p = self.num_param
        e = self.num_estimate
        i = NUM_IMPORTANCE_INGREDIENTS
        self.sampled_params = tf.Variable(tf.fill((n, p), np.nan),
                                          dtype=tf.float32)
        self.sampled_chols = tf.Variable(tf.fill((n, e, e), np.nan),
                                         dtype=tf.float32)
        self.sampled_targets = tf.Variable(tf.fill((n, i), np.nan),
                                           dtype=tf.float32)
        super().load(*args, **kwargs)
        mins, maxs = self.mins_and_maxs_valid()
        self.min_params_valid = mins
        self.min_params_valid = maxs

    @tf.function
    def assign_iteration_results(
            self,
            params,
            mean,
            cov_chol,
            inv_chol,
            chol_det,
            importance,
            sampled_params,
            sampled_targets,
            sampled_chols,
    ) -> None:

        self.params.assign(params)
        self.mean.assign(mean)
        self.cov_chol.assign(cov_chol)
        self.inv_chol.assign(inv_chol)
        self.chol_det.assign(chol_det)
        self.importance.assign(importance)

        self.sampled_params[self.iteration_num].assign(sampled_params)
        self.sampled_targets[self.iteration_num].assign(sampled_targets)
        self.sampled_chols[self.iteration_num].assign(sampled_chols)

        self.iteration_num.assign(self.iteration_num + 1)

    def generate_peripheral_samples(
            self,
            mins_valid,
            maxs_valid,
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
            (pi, ci), ti = self.generate_peripheral_samples_batch(mins_valid,
                                                                  maxs_valid)
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

        targets_peripheral, chols_peripheral = self.sample_ingredients(
            sampled_params_peripheral,
            self.peripheral_batch_size,
        )

        sim_blob = (sampled_params_peripheral, chols_peripheral)
        return sim_blob, targets_peripheral

    @tf.function
    def sampling_iteration(
            self,
    ):

        new_params = self.random_params_step(self.params, self.cov_chol)
        new_mean, new_cov_chol, new_inv_chol, new_chol_det = \
            self.sample_statistics(new_params, self.num_chains)
        prop_prob_new_given_old = self.params_proposal_pdf_proportional(
            new_params, self.params, self.inv_chol, self.chol_det,
        )
        prop_prob_old_given_new = self.params_proposal_pdf_proportional(
            self.params, new_params, new_inv_chol, new_chol_det,
        )
        new_importance_ingredients = self.importance_ingredients(new_params,
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
                                      cov_chol, inv_chol, chol_det, importance,
                                      new_params,
                                      new_importance_ingredients,
                                      new_cov_chol)

    @tf.function
    def random_params_step(
            self,
            params: Tensor2[tf32, Chains, Params],
            cov_chol: Tensor3[tf32, Chains, Estimates, Estimates],
    ) -> Tensor2[tf32, Chains, Params]:

        # TODO: Nothing currently to stop a step into an invalid param
        step = common.PARAM_MARKOV_CHAIN_STEP_SIZE

        z_unknown = tf.random.normal((self.num_chains, self.num_unknown_param))
        z_known = tf.random.normal((self.num_chains, self.num_known_param))
        params_unknown = params[:, :self.num_unknown_param]
        params_known = params[:, self.num_unknown_param:]
        new_params_unknown = (
            params_unknown +
            step * tf.linalg.matmul(cov_chol, z_unknown[:, :, None])[:, :, 0]
        )
        new_params_known = params_known + step * self.sd_known * z_known
        new_params = tf.concat([new_params_unknown, new_params_known], axis=1)

        return new_params

    @tf.function
    def sample_ingredients(
            self,
            params: Tensor2[tf32, Samples, Params],
            num_chains: int,
    ) -> Tuple[
         Tensor2[tf32, Samples, ImportanceIngredients],
         Tensor3[tf32, Samples, Estimates, Estimates],
    ]:

        mean, cov_chol, inv_chol, chol_det = self.sample_statistics(params,
                                                                    num_chains)
        importance_ingredients = self.importance_ingredients(params,
                                                             mean,
                                                             cov_chol,
                                                             chol_det)
        return importance_ingredients, cov_chol

    @tf.function
    def importance_ingredients(
            self,
            params: Tensor2[tf32, Chains, Params],
            centroid: Tensor2[tf32, Chains, Estimates],
            cov_chol: Tensor3[tf32, Chains, Estimates, Estimates],
            chol_det: Tensor1[tf32, Chains],
    ) -> Tensor2[tf32, Chains, ImportanceIngredients]:

        eps = common.SMALLEST_LOGABLE_NUMBER

        overlaps = self.overlaps_estimates_box(centroid, cov_chol)
        known_params = params[:, self.num_unknown_param:]
        known_params_valid = tf.math.reduce_all(
            (known_params >= common.PARAMS_MIN)
            &
            (known_params <= common.PARAMS_MAX),
            axis=1
        )
        known_params_valid = tf.cast(known_params_valid, tf.float32)
        importance_if_overlaps = tf.constant(1.) / chol_det

        importance_ingredients_unlog = tf.stack([
            importance_if_overlaps,
            overlaps * known_params_valid
        ], axis=1)

        return tf.math.log(importance_ingredients_unlog + eps)                 # type: ignore

    @tf.function
    def get_importance(
            self,
            importance_ingredients: Tensor2[tf32, Chains,
                                                  ImportanceIngredients]
    ) -> Tensor1[tf32, Chains]:

        return tf.math.exp(tf.reduce_sum(importance_ingredients, axis=1))

    @tf.function
    def overlaps_estimates_box(
            self,
            centroid: Tensor2[tf32, Chains, Estimates],
            cov_chol: Tensor3[tf32, Chains, Estimates, Estimates],
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

    @tf.function
    def sample_statistics(
            self,
            params: Tensor2[tf32, Chains, Params],
            num_chains: int,
    ) -> Tuple[
        Tensor2[tf32, Chains, Estimates],
        Tensor3[tf32, Chains, Estimates, Estimates],
        Tensor3[tf32, Chains, Estimates, Estimates],
        Tensor1[tf32, Chains],
    ]:

        params_repeated = tf.repeat(params, self.sample_size, axis=0)
        estimates = self.sampling_distribution_fn(params_repeated)
        estimates_grouped = tf.reshape(estimates, (num_chains,
                                                   self.sample_size,
                                                   self.num_estimate))
        xbar = tf.reduce_mean(estimates_grouped, axis=1)
        l = tfp.stats.cholesky_covariance(estimates_grouped, sample_axis=1)
        identity = tf.eye(self.num_estimate, batch_shape=(num_chains, ))
        inv_l = tf.linalg.triangular_solve(l, identity)
        det_l = tf.reduce_prod(tf.linalg.diag_part(l), axis=1)

        return xbar, l, inv_l, det_l

    @tf.function
    def covariance_cholesky_computation(
            self,
            estimates: Tensor3[tf32, Chains, Samples, Estimates],
            estimates_mean: Tensor2[tf32, Chains, Estimates],
    ) -> Tensor3[tf32, Chains, Estimates, Estimates]:

        # TODO: Check if tfp.stats.cholesky_covariance produces stable enough
        #       output consistently to remove this function.  Currently unused
        #       but not deleting as might be needed in future.

        estimates_centred = estimates - estimates_mean[:, None, :]
        c, n, p = estimates_centred.shape
        q, r = tf.linalg.qr(estimates_centred)
        r = r * tf.sign(tf.linalg.diag_part(r))[:, :, None]
        l = tf.transpose(r, perm=(0, 2, 1)) / tf.math.sqrt(n - 1.)

        return l

    @tf.function
    def params_proposal_pdf_proportional(
            self,
            x: Tensor2[tf32, Chains, Params],
            mu: Tensor2[tf32, Chains, Params],
            sigma_chol_inv: Tensor3[tf32, Chains, Estimates, Estimates],
            sigma_chol_det: Tensor1[tf32, Chains],
    ) -> Tensor1[tf32, Chains]:

        d = x - mu
        d_unknown = d[: ,:self.num_unknown_param]
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

    @tf.function
    def is_inside_support_region(
            self,
            targets: Tensor2[tf32, Samples, ImportanceIngredients],
    ) -> Tensor1[ttf.bool, Samples]:

        return targets[:, 1] >= 0.                                             # type: ignore

    @tf.function
    def get_chol_det_from_targets(
            self,
            targets: Tensor2[tf32, Samples, ImportanceIngredients],
    ) -> Tensor1[tf32, Samples]:

        return 1. / tf.math.exp(targets[:, 0])

    def get_smoothing_regions(
            self,
            targets: Tensor2[tf32, Samples, ImportanceIngredients],
            chols: Tensor3[tf32, Samples, Estimates, Estimates],
            targets_peripheral: Tensor2[tf32, Samples, ImportanceIngredients],
            chols_peripheral: Tensor3[tf32, Samples, Estimates, Estimates],
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
