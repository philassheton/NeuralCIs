from neuralcis._simulator_net_cached import _SimulatorNetCached
from neuralcis import common

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm

# typing
from typing import Callable, Tuple, Optional
from neuralcis.common import Samples, Estimates, Indices, Params, UnknownParams
from neuralcis.common import MinAndMax, ImportanceIngredients
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3
from tensor_annotations import tensorflow as ttf

from neuralcis.common import NetInputs

tf32 = ttf.float32

NetInputSimulationBlob = Tuple[
    Tensor2[tf32, Samples, Params],                           # Centroid
    Tensor3[tf32, Samples, UnknownParams, UnknownParams],     # Cholesky factor
]
NetInputBlob = Tensor2[tf32, Samples, Params]
NetOutputBlob = Tensor2[tf32, Samples, ImportanceIngredients]
NetTargetBlob = Tensor2[tf32, Samples, ImportanceIngredients]
NUM_IMPORTANCE_INGREDIENTS = 2

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
#              in general, in particular in that the peripheral zeroes will
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


class _SamplingFeelerNet(_SimulatorNetCached):
    absolute_loss_increase_tol = common.ABS_LOSS_INCREASE_TOL_FEELER_NET
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
            **network_setup_args,
    ) -> None:

        tf.keras.models.Model.__init__(self)

        self.sampling_distribution_fn = sampling_distribution_fn
        self.num_estimate = estimates_min_and_max.shape[0]
        self.estimates_min = estimates_min_and_max[:, 0]
        self.estimates_max = estimates_min_and_max[:, 1]

        # We will draw our very first param sample from the estimates box,
        #   since we are for starters assuming that the estimates are indeed
        #   estimates of the params, so that's probably a good place to start.
        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param
        self.num_param = num_unknown_param + num_known_param
        self.first_params_min = tf.concat([
            self.estimates_min,
            tf.repeat(common.PARAMS_MIN, num_known_param),
        ], axis=0)
        self.first_params_widths = tf.concat([
            self.estimates_max - self.estimates_min,
            tf.repeat(common.PARAMS_MAX - common.PARAMS_MIN, num_known_param),
        ], axis=0)

        param_nans = tf.fill(self.num_param, np.nan)
        self.min_params_simulated = tf.Variable(param_nans)
        self.max_params_simulated = tf.Variable(param_nans)

        self.sample_size = sample_size
        self.sd_known = common.KNOWN_PARAM_MARKOV_CHAIN_SD
        self.num_chains = common.FEELER_NET_NUM_CHAINS
        self.chain_length = common.FEELER_NET_MARKOV_CHAIN_LENGTH
        self.num_peripheral_samples = common.FEELER_NET_NUM_PERIPHERAL_POINTS

        self.cache_size = (self.num_chains *
                           self.chain_length +
                           self.num_peripheral_samples)

        # See note 2 at the top of this script.  This computes an adjustment
        # of 1 / .82 for a self.sample_size of 100
        self.sd_sampling_error_adjust = 1. / tf.sqrt(
            tfp.distributions.Chi2(sample_size - 1).quantile(.005)
            /
            (sample_size - 1)
        )

        # TODO: Redesign so this lives somewhere more comfortable
        super().__init__(
            cache_size=self.cache_size,
            num_outputs_for_each_net=(NUM_IMPORTANCE_INGREDIENTS,),
            instance_tf_variables_to_save=('min_params_simulated',
                                           'max_params_simulated'),
            **network_setup_args)

    def simulate_training_data_cache(
            self,
            n: int,
    ) -> Tuple[NetInputSimulationBlob, NetTargetBlob]:

        assert n == self.cache_size
        return self.simulate_param_samples(self.num_chains,
                                           self.chain_length,
                                           self.num_peripheral_samples,
                                           store_min_and_max=True)

    def simulate_validation_data_cache(
            self,
    ) -> Tuple[NetInputBlob, Optional[NetTargetBlob]]:

        (params_sampled, chols), targets = self.simulate_param_samples(
            common.FEELER_NET_VALIDATION_SET_NUM_CHAINS,
            common.FEELER_NET_VALIDATION_SET_MARKOV_CHAIN_LENGTH,
            common.FELLER_NET_VALIDATION_SET_NUM_PERIPHERAL_SAMPLES
        )
        return params_sampled, targets

    def simulate_fake_training_data(
            self,
            n: ttf.int32,
    ) -> Tuple[NetInputBlob, Optional[NetTargetBlob]]:

        net_input_blob = tf.zeros((n, self.num_param))
        net_target_blob = tf.zeros((n, NUM_IMPORTANCE_INGREDIENTS))
        return net_input_blob, net_target_blob

    def simulate_param_samples(
            self,
            num_chains: int,
            chain_length: int,
            num_peripheral_samples: int,
            store_min_and_max: bool = False,
    ) -> Tuple[NetInputSimulationBlob, NetTargetBlob]:

        params_sampled = []
        chols_sampled = []
        targets = []
        print("Generating chains:")
        for _ in tqdm(range(num_chains)):
            (p, c), t = self.simulate_single_chain(chain_length)
            params_sampled.append(p)
            chols_sampled.append(c)
            targets.append(t)

        params_sampled = tf.concat(params_sampled, axis=0)
        chols_sampled = tf.concat(chols_sampled, axis=0)
        targets = tf.concat(targets, axis=0)

        # Generate extra samples around the edges that force the
        #    probability of assigning non-zero probability at the edges
        #    down to zero.
        # TODO: Make this fit more snugly to the countours of the original
        #       sample.
        print(f"Generating {num_peripheral_samples} peripheral samples.")
        valid_rows = tf.where(self.is_inside_support_region(targets))[:, 0]
        params_sampled_valid = tf.gather(params_sampled, valid_rows, axis=0)
        mins = tf.reduce_min(params_sampled_valid, axis=0)
        maxs = tf.reduce_max(params_sampled_valid, axis=0)
        diffs = maxs - mins
        u = tf.random.uniform((num_peripheral_samples, self.num_param))
        params_sampled_peripheral = u * diffs[None, :] + mins[None, :]

        print("... generating target values")
        targets_peripheral, chols_peripheral = self.sample_ingredients_batch(
            params_sampled_peripheral
        )
        print("... done.")

        # Compute for each sample a region around the sample that can be
        # substituted for that sample in order to smooth the surface
        chols_sampled, chols_peripheral = self.compute_smoothing_regions(
            targets, chols_sampled,
            targets_peripheral, chols_peripheral,
            mins, maxs,
        )

        params_sampled = tf.concat([params_sampled, params_sampled_peripheral],
                                   axis=0)
        chols = tf.concat([chols_sampled, chols_peripheral], axis=0)
        targets = tf.concat([targets, targets_peripheral], axis=0)

        if store_min_and_max:
            self.min_params_simulated.assign(mins)
            self.max_params_simulated.assign(maxs)

        return (params_sampled, chols), targets

    def simulate_single_chain(
            self,
            chain_length: int,
    ) -> Tuple[NetInputSimulationBlob, NetTargetBlob]:

        u = tf.random.uniform((self.num_param,))
        old_params = u * self.first_params_widths + self.first_params_min
        old_mean, old_cov_chol, old_inv_chol, old_chol_det = \
            self.sample_statistics(old_params)
        importance_ingredients = self.importance_ingredients(old_params,
                                                             old_mean,
                                                             old_cov_chol,
                                                             old_chol_det)
        old_importance = tf.math.exp(tf.reduce_sum(importance_ingredients))

        param_samples = tf.TensorArray(tf.float32,
                                       size=chain_length,
                                       element_shape=(self.num_param,))
        chols = tf.TensorArray(tf.float32,
                               size=chain_length,
                               element_shape=(self.num_estimate,
                                              self.num_estimate))
        targets = tf.TensorArray(tf.float32,
                                 size=chain_length,
                                 element_shape=(NUM_IMPORTANCE_INGREDIENTS,))
        param_samples = param_samples.write(0, old_params)
        chols = chols.write(0, old_cov_chol)
        targets = targets.write(0, importance_ingredients)

        for i in range(chain_length - 1):
            (
                new_params, new_cov_chol, importance_ingredients,
                old_params, old_mean, old_cov_chol, old_inv_chol,
                old_chol_det, old_importance,
            ) = self.sampling_iteration(
                old_params,
                old_mean, old_cov_chol, old_inv_chol, old_chol_det,
                old_importance,
            )
            param_samples = param_samples.write(i+1, new_params)
            chols = chols.write(i+1, new_cov_chol)
            targets = targets.write(i+1, importance_ingredients)

        return (param_samples.stack(), chols.stack()), targets.stack()

    @tf.function
    def sampling_iteration(
            self,
            old_params,
            old_mean,
            old_cov_chol,
            old_inv_chol,
            old_chol_det,
            old_importance,
    ):

        new_params = self.random_params_step(old_params, old_cov_chol)
        new_mean, new_cov_chol, new_inv_chol, new_chol_det = \
            self.sample_statistics(new_params)
        prop_prob_new_given_old = self.params_proposal_pdf_proportional(
            new_params, old_params, old_inv_chol, old_chol_det,
        )
        prop_prob_old_given_new = self.params_proposal_pdf_proportional(
            old_params, new_params, new_inv_chol, new_chol_det,
        )
        new_importance_ingredients = self.importance_ingredients(new_params,
                                                                 new_mean,
                                                                 new_cov_chol,
                                                                 new_chol_det)
        new_importance = tf.math.exp(tf.reduce_sum(new_importance_ingredients))

        # This is not quite really a proper importance sample
        # ...we just use the MCMC to help us decide which way to walk next
        acceptance_prob = tf.minimum(
            1.,
            (new_importance * prop_prob_old_given_new) /
            (old_importance * prop_prob_new_given_old),
        )
        accepted = acceptance_prob > tf.random.uniform(())

        old_params = tf.where(accepted, new_params, old_params)
        old_mean = tf.where(accepted, new_mean, old_mean)
        old_cov_chol = tf.where(accepted, new_cov_chol, old_cov_chol)
        old_inv_chol = tf.where(accepted, new_inv_chol, old_inv_chol)
        old_chol_det = tf.where(accepted, new_chol_det, old_chol_det)
        old_importance = tf.where(accepted, new_importance, old_importance)

        return (
            new_params,
            new_cov_chol,
            new_importance_ingredients,
            old_params,
            old_mean,
            old_cov_chol,
            old_inv_chol,
            old_chol_det,
            old_importance,
        )

    @tf.function
    def pick_indices_from_cache(
            self,
            cache: Tuple[NetInputSimulationBlob, NetTargetBlob],
            indices: Tensor1[ttf.int16, Indices],
    ) -> Tuple[NetInputSimulationBlob, NetTargetBlob]:

        (param_samples_cache, chols_cache), targets_cache = cache
        param_samples = tf.gather(param_samples_cache, indices)
        chols = tf.gather(chols_cache, indices)
        targets = tf.gather(targets_cache, indices)

        return (param_samples, chols), targets

    @tf.function
    def simulate_data_from_cache_chunk(
            self,
            input_simulation_blob: NetInputSimulationBlob,
            target_blob: NetTargetBlob,
    ) -> NetInputBlob:

        # TODO: I think it makes the most sense not to smooth the known_params,
        #       now that we condition on it, and since we have no info about
        #       sensible covariances.  (Hence using tf.zeros below).
        #       Look further into this later.
        (param_samples, chols) = input_simulation_blob
        n, _ = param_samples.shape
        z = tf.random.normal((n, self.num_unknown_param, 1))
        smoothing_unknown = tf.linalg.matmul(chols, z)[:, :, 0]
        smoothing_known = tf.zeros((n, self.num_known_param))
        smoothing = tf.concat([smoothing_unknown, smoothing_known], axis=1)
        param_samples = param_samples + smoothing

        return param_samples, target_blob

    @tf.function
    def random_params_step(
            self,
            params: Tensor1[tf32, Params],
            cov_chol: Tensor2[tf32, Estimates, Estimates],
    ) -> Tensor1[tf32, Params]:

        step = common.PARAM_MARKOV_CHAIN_STEP_SIZE

        z_unknown = tf.random.normal((self.num_unknown_param,))
        z_known = tf.random.normal((self.num_known_param,))
        params_unknown = params[:self.num_unknown_param]
        params_known = params[self.num_unknown_param:]
        new_params = tf.concat([
            params_unknown + step * tf.linalg.matvec(tf.transpose(cov_chol),
                                                     z_unknown),
            params_known + step * self.sd_known * z_known,
        ], axis=0)

        return new_params

    def sample_ingredients_batch(
            self,
            params: Tensor2[tf32, Samples, Params],
    ) -> Tuple[
        Tensor2[tf32, Samples, ImportanceIngredients],
        Tensor3[tf32, Samples, Estimates, Estimates],
    ]:

        # TODO: Should be possible to speed this up by making the whole
        #       sampling process properly vectorized.
        return tf.map_fn(self.sample_ingredients, params,
                         dtype = (tf.float32, tf.float32))

    @tf.function
    def sample_ingredients(
            self,
            params: Tensor1[tf32, Params],
    ) -> Tuple[
         Tensor1[tf32, ImportanceIngredients],
         Tensor2[tf32, Estimates, Estimates],
    ]:

        mean, cov_chol, inv_chol, chol_det = self.sample_statistics(params)
        importance_ingredients = self.importance_ingredients(params,
                                                             mean,
                                                             cov_chol,
                                                             chol_det)
        return importance_ingredients, cov_chol

    @tf.function
    def importance_ingredients(
            self,
            params: Tensor1[tf32, Params],
            centroid: Tensor1[tf32, Estimates],
            cov_chol: Tensor2[tf32, Estimates, Estimates],
            chol_det: Tensor0[tf32],
    ) -> Tensor1[tf32, ImportanceIngredients]:

        eps = common.SMALLEST_LOGABLE_NUMBER

        overlaps = self.overlaps_estimates_box(centroid, cov_chol)
        known_params = params[self.num_unknown_param:]
        known_params_valid = tf.math.reduce_all(
            (known_params >= common.PARAMS_MIN)
            &
            (known_params <= common.PARAMS_MAX)
        )
        known_params_valid = tf.cast(known_params_valid, tf.float32)
        importance_if_overlaps = tf.constant(1.) / chol_det

        importance_ingredients_unlog = tf.stack([
            importance_if_overlaps,
            overlaps * known_params_valid
        ])

        return tf.math.log(importance_ingredients_unlog + eps)                 # type: ignore

    @tf.function
    def overlaps_estimates_box(
            self,
            centroid: Tensor1[tf32, Estimates],
            cov_chol: Tensor2[tf32, Estimates, Estimates],
    ) -> Tensor0[tf32]:

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
        sds = tf.sqrt(tf.reduce_sum(tf.square(cov_chol), axis=1))
        bounding_box_lower = centroid - cutoff * sds
        bounding_box_upper = centroid + cutoff * sds
        overlaps = (
                tf.math.reduce_all(bounding_box_lower <= self.estimates_max)
                &
                tf.math.reduce_all(bounding_box_upper >= self.estimates_min)
        )

        return tf.cast(overlaps, tf.float32)

    @tf.function
    def sample_statistics(
            self,
            params: Tensor1[tf32, Params],
    ) -> Tuple[
        Tensor1[tf32, Estimates],
        Tensor2[tf32, Estimates, Estimates],
        Tensor2[tf32, Estimates, Estimates],
        Tensor0[tf32],
    ]:

        params_repeated = tf.tile(params[None, :], (self.sample_size, 1))
        estimates = self.sampling_distribution_fn(params_repeated)
        xbar = tf.reduce_mean(estimates, axis=0)
        l = self.covariance_cholesky_computation(estimates)
        det_l = tf.reduce_prod(tf.linalg.diag_part(l))
        inv_l = tf.linalg.triangular_solve(l, tf.eye(self.num_estimate))

        return xbar, l, inv_l, det_l                                           # type: ignore

    @tf.function
    def covariance_cholesky_computation(
            self,
            estimates: Tensor2[tf32, Samples, Estimates]
    ) -> Tensor2[tf32, Estimates, Estimates]:

        # TODO: This should be more stable, and possibly slightly faster at
        #       higher dimensions than estimating the covariance matrix and
        #       then Cholesky decomposing.  Need to investigate at higher
        #       dimensions whether this is actually better.
        estimates_centred = estimates - tf.reduce_mean(estimates,
                                                       axis=0,
                                                       keepdims=True)

        n, p = estimates_centred.shape
        q, r = tf.linalg.qr(estimates_centred)
        r = r * tf.sign(tf.linalg.diag_part(r))[:, None]
        l = tf.transpose(r) / tf.math.sqrt(n - 1.)

        return l

    @tf.function
    def params_proposal_pdf_proportional(
            self,
            x: Tensor1[tf32, Params],
            mu: Tensor1[tf32, Params],
            sigma_chol_inv: Tensor2[tf32, Estimates, Estimates],
            sigma_chol_det: Tensor0[tf32],
    ) -> Tensor0[tf32]:

        d = x - mu
        d_unknown = d[:self.num_unknown_param]
        d_known = d[self.num_unknown_param:]

        z_unknown = tf.linalg.matvec(tf.transpose(sigma_chol_inv), d_unknown)
        z_known = d_known / self.sd_known
        z = tf.concat([z_unknown, z_known], axis=0)
        z_norm_sq = tf.square(tf.norm(z))

        det_unknown = sigma_chol_det
        det_known = tf.math.pow(self.sd_known, self.num_known_param)
        return tf.math.exp(-0.5 * z_norm_sq) / (det_unknown * det_known)       # NB: We do *not* need sqrt on the determinant because it is the sqrt of the Cholesky factor (already sqrted)

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

    def compute_smoothing_regions(
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
        tf.print(f"Cholesky scale factor: {chol_scale_factor}.  If this is "
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

    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: Optional[NetTargetBlob] = None,
    ) -> Tensor0[tf32]:

        tf.debugging.check_numerics(net_outputs, "NaN in outputs!")
        tf.debugging.check_numerics(target_outputs, "NaN in targets!")
        return tf.reduce_mean(tf.square(net_outputs - target_outputs))         # type: ignore

    @tf.function
    def net_inputs(
            self,
            inputs: NetInputBlob,
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], ...]:

        # TODO: can we just push this up to the _SimulatorNet?
        return inputs,                                                         # type: ignore

    @tf.function
    def get_log_importance_from_net(
            self,
            params: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        importance_ingredients = self.call_tf(params)
        importance_log = tf.reduce_sum(importance_ingredients, axis=1)

        # Add a punitive amount for being outside the region sampled from
        param_too_low_by = tf.maximum(
            self.min_params_simulated[None, :] - params, 0.
        )
        param_too_high_by = tf.maximum(
            params - self.max_params_simulated[None, :], 0.
        )
        param_out_of_bound_by = tf.maximum(param_too_low_by, param_too_high_by)
        greatest_out_of_bound = tf.reduce_max(param_out_of_bound_by, axis=1)

        # oob = out of bounds
        oob = tf.sign(greatest_out_of_bound)
        not_oob = 1. - oob

        log_eps = tf.math.log(common.SMALLEST_LOGABLE_NUMBER)
        out_of_bounds_val = log_eps - greatest_out_of_bound

        return not_oob*importance_log + oob*out_of_bounds_val
