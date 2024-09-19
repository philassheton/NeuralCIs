from neuralcis._simulator_net_cached import _SimulatorNetCached
from neuralcis._sampling_feeler_generator import _SamplingFeelerGenerator
from neuralcis import common

import tensorflow as tf
import numpy as np

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


class _SamplingFeelerNet(_SimulatorNetCached):
    relative_loss_increase_tol = common.REL_LOSS_INCREASE_TOL_FEELER_NET

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
            **network_setup_args,
    ) -> None:

        cache_size = (num_chains * chain_length +
                      num_peripheral_batches * peripheral_batch_size)

        feeler_data_generator = _SamplingFeelerGenerator(
            estimates_min_and_max,
            sampling_distribution_fn,
            num_unknown_param,
            num_known_param,
            sample_size,
            sd_known,
            num_chains,
            chain_length,
            peripheral_batch_size,
            num_peripheral_batches,
        )

        super().__init__(
            cache_size=cache_size,
            num_inputs_for_each_net=(num_unknown_param + num_known_param,),
            num_outputs_for_each_net=(NUM_IMPORTANCE_INGREDIENTS,),
            subobjects_to_save=({"feelergen": feeler_data_generator}),
            instance_tf_variables_to_save=('min_params_valid',
                                           'max_params_valid'),
            **network_setup_args
        )

        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param
        self.num_param = num_unknown_param + num_known_param

        self.feeler_data_generator = feeler_data_generator

        self.min_params_valid = tf.Variable(tf.fill((self.num_param,), np.nan))
        self.max_params_valid = tf.Variable(tf.fill((self.num_param,), np.nan))

    def simulate_training_data_cache(
            self,
            n: int,
    ) -> Tuple[NetInputSimulationBlob, NetTargetBlob]:

        assert n == self.cache_size
        self.feeler_data_generator.fit()

        mins, maxs = self.feeler_data_generator.mins_and_maxs_valid()
        self.min_params_valid.assign(mins)
        self.max_params_valid.assign(maxs)

        sim_blob, target_blob = self.get_data_from_generator()

        return sim_blob, target_blob

    def get_data_from_generator(
            self
    ) -> Tuple[NetInputSimulationBlob, NetTargetBlob]:

        sim_blob = (self.feeler_data_generator.sampled_params,
                    self.feeler_data_generator.sampled_chols)
        target_blob = self.feeler_data_generator.sampled_targets
        return sim_blob, target_blob

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
        param_samples = param_samples + smoothing

        return param_samples, target_blob

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
        return inputs,                                                         # type: ignore

    @tf.function
    def get_log_importance_from_net(
            self,
            params: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        importance_ingredients = self.call_tf(params)
        importance_log = tf.reduce_sum(importance_ingredients, axis=1)

        # Add a punitive amount for being outside the region sampled from
        param_too_low_by = tf.maximum(self.min_params_valid[None, :] - params,
                                      0.)
        param_too_high_by = tf.maximum(params - self.max_params_valid[None, :],
                                       0.)
        param_out_of_bound_by = tf.maximum(param_too_low_by, param_too_high_by)
        greatest_out_of_bound = tf.reduce_max(param_out_of_bound_by, axis=1)

        # oob = out of bounds
        oob = tf.sign(greatest_out_of_bound)
        not_oob = 1. - oob

        log_eps = tf.math.log(common.SMALLEST_LOGABLE_NUMBER)
        out_of_bounds_val = log_eps - greatest_out_of_bound

        return not_oob*importance_log + oob*out_of_bounds_val
