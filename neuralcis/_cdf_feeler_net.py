from neuralcis._simulator_net_cached import _SimulatorNetCached
from neuralcis._cdf_feeler_generator import _CDFFeelerGenerator
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
NUM_CDF_MEASURES = 2


class _CDFFeelerNet(_SimulatorNetCached):
    relative_loss_increase_tol = common.REL_LOSS_INCREASE_TOL_FEELER_NET

    def __init__(
            self,
            num_unknown_param: int,
            num_known_param: int,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],  # params
                Tensor2[tf32, Samples, Estimates],  # -> ys
            ],
            sampling_feeler_generator: _SamplingFeelerGenerator,
            p_fn: Callable[
                [Tensor2[tf32, Samples, Estimates],
                 Tensor2[tf32, Samples, Params]],
                Tensor1[tf32, Samples]
            ],
            num_samples_per_param: int = \
                                      common.CDF_FEELER_SAMPLES_PER_TEST_PARAM,
            num_params_per_batch: int = common.CDF_FEELER_PARAMS_PER_BATCH,
            subsample_proportion: float = 1.,
            **network_setup_args,
    ) -> None:

        cdf_feeler_generator = _CDFFeelerGenerator(
            sampling_distribution_fn,
            sampling_feeler_generator,
            p_fn,
            num_samples_per_param,
            num_params_per_batch,
            subsample_proportion,
        )

        super().__init__(


            # PHIL!!
            cache_size=None,



            num_inputs_for_each_net=(num_unknown_param + num_known_param,),
            num_outputs_for_each_net=(NUM_CDF_MEASURES,),
            subobjects_to_save=({"cdfgen": cdf_feeler_generator}),
            **network_setup_args
        )

        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param
        self.num_param = num_unknown_param + num_known_param

        self.cdf_feeler_generator = cdf_feeler_generator

    def simulate_training_data_cache(
            self,
            n: int,
    ) -> Tuple[NetInputSimulationBlob, NetTargetBlob]:



        # PHIL!!  we just need to totally get rid of n and cache size
        #         from this altogether!!  It can just be left to generate
        #         whatever size cache it likes, and then we set self.cache_size
        #         based on that.
        assert n is None





        # TODO: Shouldn't even be an n here.
        # assert n == self.cache_size
        self.cdf_feeler_generator.fit()
        sim_blob, target_blob = self.get_data_from_generator()




        # PHIL!!  This should maybe be returned rather than directly stored
        #         in here.
        #          -> Tuple[int, NetInputSimulationBlob, NetTargetBlob]
        self.cache_size = self.cdf_feeler_generator.num_param_samples



        return sim_blob, target_blob

    def get_data_from_generator(
            self
    ) -> Tuple[NetInputSimulationBlob, NetTargetBlob]:

        generator = self.cdf_feeler_generator
        sampling_feeler_indices = generator.sampling_feeler_indices

        params = generator.sampling_feeler_generator.sampled_params
        chols = generator.sampling_feeler_generator.sampled_chols

        params = tf.gather(params, sampling_feeler_indices, axis=0)
        chols = tf.gather(chols, sampling_feeler_indices, axis=0)

        targets = tf.stack([
            generator.darlings,
            generator.ks,
        ], axis=1)

        sim_blob = (params, chols)
        return sim_blob, targets

    # TODO: Everything beneath here is copied from _SamplingFeelerNet
    #       Can we factor things so we don't need to reproduce?
    #       Perhaps make an abstract base class that has the core things in.

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

