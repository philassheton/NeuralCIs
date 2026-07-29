from ._data_saver import _DataSaver
from .common import FULL
from ._utils import split_unknown_and_known_params
from ._utils import concat_unknown_and_known_params
from . import common

import tensorflow as tf
import numpy as np
import gc
from datetime import datetime
from tqdm import tqdm
from abc import ABC, abstractmethod

# typing
from typing import Optional, Callable, Tuple
from .common import Samples, Stats, Params, UnknownParams, KnownParams
from .common import ImportanceIngredients, Chains
from tensor_annotations.tensorflow import Tensor1, Tensor2, Tensor3, Tensor4
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
#              estimates.
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
#           3) Just using bounding boxes to test for collision is pretty bad
#              regarding curse of dimensionality as the corners of the
#              hypercubes dominate space.
#
###############################################################################



class _SamplingFeelerGenerator(_DataSaver, ABC):
    smallest_profile_found_in = FULL

    def __init__(
            self,
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
            sample_size: int,
            sd_known: float,
            num_chains: int,
            chain_length: int,
            peripheral_batch_size: int,
            num_peripheral_batches: int,
            regularize_jitter_multiply: float,
            regularize_jitter_add: float,
    ):

        _DataSaver.__init__(
            self,
            instance_tf_variables_to_save=("sampled_params",
                                           "sampled_chols",
                                           "sampled_targets",
                                           "iteration_num"),
        )

        if self._skip_when_profile(profile):
            return

        self.sampling_distribution_fn = sampling_distribution_fn
        self.preprocess_params_fn = preprocess_params_fn
        self.params_is_valid_fn = params_is_valid_fn
        self.estimates_fn = estimates_fn

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

        # Set up all the tf.Variables that will be used to construct the chains
        def state_variable(shape_inner, dtype=tf.float32):
            shape = [self.num_chains] + list(shape_inner)
            nans = tf.fill(shape, np.nan)
            var = tf.Variable(nans, dtype=dtype)
            return var

        num_param = self.num_param
        num_estimate = self.num_estimate()

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

    @abstractmethod
    def sample_statistics(
            self,
            params: Tensor2[tf32, Chains, Params],
    ) -> Tuple[
        Tensor2[tf32, Chains, Params],                                         # Params (preprocessed)
        Tensor2[tf32, Chains, UnknownParams],                                  # Mean
        Tensor3[tf32, Chains, UnknownParams, UnknownParams],                   # L = Cholesky(Covariance matrix)
        Tensor3[tf32, Chains, UnknownParams, UnknownParams],                   # L^-1
        Tensor1[tf32, Chains],                                                 # |L|
        Tensor1[tf32, Chains],                                                 # Sample blob overlaps interest region
    ]:

        pass

    @abstractmethod
    def sample_starting_params(
            self, n: int,
    ) -> Tensor2[tf32, Chains, Params]:

        pass

    def fit(self, *args, **kwargs) -> None:
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

    def num_estimate(self):
        return self.num_unknown_param

    def initialise_for_training(self):
        self.iteration_num.assign(0)
        params = self.sample_starting_params(self.num_chains)

        params_pp, mean, cov_chol, inv_chol, chol_det, overlaps = \
            self.sample_statistics(params)
        importance_ingredients = self.importance_ingredients(params_pp,
                                                             chol_det,
                                                             overlaps)
        importance = self.get_importance(importance_ingredients)

        # We want to store un-preprocessed params as current state from which
        #   to step from (this way we will naturally diffuse across different
        #   discrete possibilities) but we want to store for the long term the
        #   discretized value params_pp (pp=preprocessed)_ for training our
        #   nets with.
        self.assign_iteration_results(params, mean,
                                      cov_chol, inv_chol, chol_det, importance)

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
            (num_samples, self.num_estimate(), self.num_estimate()),
        )
        self.sampled_targets = tf.reshape(
            targets,
            (num_samples, NUM_IMPORTANCE_INGREDIENTS),
        )

    @tf.function(jit_compile=False)
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
                           self.num_estimate(),
                           self.num_estimate()),
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

    def sampling_iteration(
            self,
    ) -> Tuple[Tensor2[tf32, Chains, Params],
               Tensor3[tf32, Chains, UnknownParams, UnknownParams],
               Tensor2[tf32, Chains, ImportanceIngredients]]:

        new_params = self.random_params_step(self.params, self.cov_chol)
        (new_params_pp,
         new_mean, new_cov_chol, new_inv_chol, new_chol_det,
         new_overlaps) = self.sample_statistics(new_params)
        prop_prob_new_given_old = self.params_proposal_pdf_proportional(
            new_params, self.params, self.inv_chol, self.chol_det,
        )
        prop_prob_old_given_new = self.params_proposal_pdf_proportional(
            self.params, new_params, new_inv_chol, new_chol_det,
        )
        new_importance_ingredients = self.importance_ingredients(
            new_params_pp,
            new_chol_det,
            new_overlaps
        )
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

        params_unknown, params_known = split_unknown_and_known_params(
            params,
            self.num_unknown_param,
            self.num_known_param,
        )

        new_params_unknown = (
            params_unknown +
            tf.linalg.matmul(cov_chol, z_unknown[:, :, None])[:, :, 0]
        )
        new_params_known = params_known + self.sd_known * z_known
        new_params = concat_unknown_and_known_params(new_params_unknown,
                                                     new_params_known)

        return new_params

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

        for i in tqdm(range(self.num_peripheral_batches)):
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

    def sample_ingredients(
            self,
            params: Tensor2[tf32, Samples, Params],
    ) -> Tuple[
        Tensor2[tf32, Chains, Params],
        Tensor3[tf32, Chains, UnknownParams, UnknownParams],
        Tensor2[tf32, Chains, ImportanceIngredients],
    ]:

        params_preproc, mean, cov_chol, inv_chol, chol_det, hits_inner = \
            self.sample_statistics(params)
        importance_ingredients = self.importance_ingredients(params_preproc,
                                                             chol_det,
                                                             hits_inner)
        return params_preproc, cov_chol, importance_ingredients

    def importance_ingredients(
            self,
            params: Tensor2[tf32, Chains, Params],
            chol_det: Tensor1[tf32, Chains],
            overlaps: Tensor1[tf32, Chains],
    ) -> Tensor2[tf32, Chains, ImportanceIngredients]:

        eps = common.SMALLEST_LOGABLE_NUMBER
        importance_if_overlaps = tf.constant(1.) / chol_det

        # For those params outside of valid ranges, we keep the samples, so we
        #   can learn not to generate them, and zero out their importance and
        #   overlaps values, in case those are NaN values.
        params_valid_each = self.params_is_valid_fn(params)
        params_valid_all = tf.math.reduce_all(params_valid_each, axis=1)
        overlaps = tf.where(params_valid_all, overlaps, 0.0)
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

    def params_proposal_pdf_proportional(
            self,
            x: Tensor2[tf32, Chains, Params],
            mu: Tensor2[tf32, Chains, Params],
            sigma_chol_inv: Tensor3[tf32, Chains, UnknownParams,
                                                  UnknownParams],
            sigma_chol_det: Tensor1[tf32, Chains],
    ) -> Tensor1[tf32, Chains]:

        d = x - mu
        d_unknown, d_known = split_unknown_and_known_params(
            d,
            self.num_unknown_param,
            self.num_known_param,
        )

        z_unknown = tf.linalg.matmul(sigma_chol_inv,
                                     d_unknown[:, :, None])[:, :, 0]
        z_known = d_known / self.sd_known
        z = concat_unknown_and_known_params(z_unknown, z_known)
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

    def min_max_supported(
            self
    ) -> Tuple[Tensor1[tf32, Params],
               Tensor1[tf32, Params]]:

        # TODO: May be faster with tf.boolean_mask rather than gather(where())?
        supported_rows = tf.where(
            self.is_inside_support_region(self.sampled_targets)
        )[:, 0]
        params_sampled_supported = tf.gather(self.sampled_params,
                                             supported_rows, axis=0)
        mins_sampled = tf.reduce_min(params_sampled_supported, axis=0)
        maxs_sampled = tf.reduce_max(params_sampled_supported, axis=0)

        return mins_sampled, maxs_sampled

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
            e = self.num_estimate()
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
