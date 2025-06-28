from ._simulator_net_cached import _SimulatorNetCached
from ._sampling_feeler_generator import _SamplingFeelerGenerator
from ._outer_feeler_generator import _OuterFeelerGenerator
from ._sampling_feeler_generator import NUM_IMPORTANCE_INGREDIENTS
from .common import FULL
from . import common

import tensorflow as tf
import numpy as np

# typing
from typing import Tuple, Optional, Union
from .common import Samples, Indices, Params, UnknownParams, NetInputs
from .common import ImportanceIngredients
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3
from tensor_annotations import tensorflow as ttf

tf32 = ttf.float32

NetInputSimulationBlob = Tuple[
    Tensor2[tf32, Samples, Params],                           # Centroid
    Tensor3[tf32, Samples, UnknownParams, UnknownParams],     # Cholesky factor
]
NetInputBlob = Tuple[Tensor2[tf32, Samples, Params],          # for log vol
                     Tensor2[tf32, Samples, Params]]          # for intersect
NetOutputBlob = Tensor2[tf32, Samples, ImportanceIngredients]
NetTargetBlob = Tensor2[tf32, Samples, ImportanceIngredients]


class _SamplingFeelerNet(_SimulatorNetCached):
    relative_loss_increase_tol = common.REL_LOSS_INCREASE_TOL_FEELER_NET
    smallest_profile_found_in = FULL

    def __init__(
            self,
            feeler_data_generator: Union[_SamplingFeelerGenerator,
                                         _OuterFeelerGenerator],
            num_unknown_param: int,
            num_known_param: int,
            profile: str,
            include_threshold: float,
            include_boost: float = 1.,
            **network_setup_args,
    ) -> None:

        super().__init__(
            profile=profile,
            num_inputs_for_each_net=(num_unknown_param + num_known_param,
                                     num_unknown_param + num_known_param),
            num_outputs_for_each_net=(1, NUM_IMPORTANCE_INGREDIENTS - 1),
            instance_tf_variables_to_save=('min_params_supported',
                                           'max_params_supported'),
            **network_setup_args
        )

        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param
        self.num_param = num_unknown_param + num_known_param

        self.feeler_data_generator = feeler_data_generator

        self.min_params_supported = tf.Variable(tf.fill((self.num_param,), np.nan))
        self.max_params_supported = tf.Variable(tf.fill((self.num_param,), np.nan))
        self.include_threshold = include_threshold
        self.include_boost = include_boost

    def simulate_training_data_cache(
            self,
    ) -> Tuple[Tuple[NetInputSimulationBlob, NetTargetBlob],
               Tensor1[ttf.int64, Indices]]:

        mins, maxs = self.feeler_data_generator.min_max_supported()
        self.min_params_supported.assign(mins)
        self.max_params_supported.assign(maxs)

        sim_blob, target_blob, indices = self.get_data_from_generator()

        return (sim_blob, target_blob), indices

    def get_data_from_generator(
            self
    ) -> Tuple[NetInputSimulationBlob,
               NetTargetBlob,
               Tensor1[ttf.int64, Indices]]:

        sim_blob = (self.feeler_data_generator.sampled_params,
                    self.feeler_data_generator.sampled_chols)
        target_blob = self.feeler_data_generator.sampled_targets

        non_nan_indices = tf.where(~tf.math.is_nan(target_blob[:, 0]))[:, 0]
        print(f"{len(non_nan_indices)} / {target_blob.shape[0]} were not NaN!")

        # We will remove any cases where known params are outside of range,
        #   since they are anyway controlled to be within range during the
        #   main simulation.  But we need to keep other params that are out of
        #   range, so that we can learn how to NOT generate those in the main
        #   simulation.
        known = self.num_known_param
        knowns_are_valid_each = self.feeler_data_generator.params_is_valid_fn(
            sim_blob[0][:, -known:], known_params_only=True,
        )
        knowns_are_valid_all = tf.reduce_all(knowns_are_valid_each, axis=1)
        knowns_valid_indices = tf.where(knowns_are_valid_all)[:, 0]
        print(f"{len(knowns_valid_indices)} / {sim_blob[0].shape[0]}"
              f" were inside known params range!")

        indices = tf.sparse.to_dense(
            tf.sets.intersection(non_nan_indices[None, :],
                                 knowns_valid_indices[None, :])
        )[0, :]

        return sim_blob, target_blob, indices

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
    ) -> Tuple[NetInputBlob, NetTargetBlob]:

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

        smoothing_log_vol = common.IMPORTANCE_INGREDIENTS_VOLUMES_SMOOTHING
        smoothing_include = \
            common.IMPORTANCE_INGREDIENTS_SHOULD_SAMPLE_SMOOTHING
        param_samples_log_vol = param_samples + smoothing_log_vol * smoothing
        param_samples_include = param_samples + smoothing_include * smoothing

        return (param_samples_log_vol, param_samples_include), target_blob

    @tf.function
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
        tf.debugging.check_numerics(inputs, "NaN in inputs!")
        return inputs                                                          # type: ignore

    @tf.function
    def get_log_importance_from_net(
            self,
            params: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        importance_ingredients = self.call_tf((params, params))
        vol = common.IMPORTANCE_INGREDIENTS_VOLUMES_INDEX
        include = common.IMPORTANCE_INGREDIENTS_SHOULD_SAMPLE_INDEX

        # TODO: Since we are now using an arbitrary log max value, this should
        #       instead be switched to be a non-logged value so that we can
        #       interpret our threshold as a probability of overlapping.
        include = tf.minimum(importance_ingredients[:, include],
                             self.include_threshold) * self.include_boost
        importance_log = (include + importance_ingredients[:, vol])            # type: ignore

        # Add a punitive amount for being outside the region sampled from
        param_too_low_by = tf.maximum(self.min_params_supported[None, :] - params,
                                      0.)
        param_too_high_by = tf.maximum(params - self.max_params_supported[None, :],
                                       0.)
        param_out_of_bound_by = tf.maximum(param_too_low_by, param_too_high_by)
        greatest_out_of_bound = tf.reduce_max(param_out_of_bound_by, axis=1)

        # oob = out of bounds
        oob = tf.sign(greatest_out_of_bound)
        not_oob = 1. - oob

        # TODO: This could be made much cleaner.  (Only exists to increase the
        #       gap between bottom and top when the threshold is very low)
        log_eps = tf.math.log(common.SMALLEST_LOGABLE_NUMBER) * self.include_boost
        out_of_bounds_val = log_eps - greatest_out_of_bound

        return not_oob*importance_log + oob*out_of_bounds_val
