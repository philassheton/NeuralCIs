from ._sampling_feeler_generator import _SamplingFeelerGenerator
from ._sampling_feeler_net import _SamplingFeelerNet
from ._outer_feeler_generator import _OuterFeelerGenerator
from ._param_sampling_net import _ParamSamplingNet
from ._is_inside_net import _IsInsideNet
from ._data_saver import _DataSaver
from .common import TESTING
from . import common

import tensorflow as tf

import tensor_annotations.tensorflow as ttf
from tensor_annotations.tensorflow import Tensor1, Tensor2, float32 as tf32
from .common import Samples
from .common import Stats, Params, UnknownParams, KnownParams, MinAndMax
from typing import Optional, Callable, Sequence, Union


class _ParamSampler(_DataSaver):
    smallest_profile_found_in = TESTING
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
            num_stat: int,
            num_unknown_param: int,
            num_known_param: int,
            known_param_indices: Sequence[int],
            profile: str,
            sample_size: int = common.SAMPLES_PER_TEST_PARAM,
            sd_known: float = common.KNOWN_PARAM_MARKOV_CHAIN_SD,
            num_chains: int = common.FEELER_NET_NUM_CHAINS,
            chain_length: int = common.FEELER_NET_MARKOV_CHAIN_LENGTH,
            peripheral_batch_size_inner: int =
                                    common.FEELER_NET_PERIPHERAL_BATCH_SIZE,
            num_peripheral_batches_inner: int =
                                    common.FEELER_NET_PERIPHERAL_BATCHES,
            peripheral_batch_size_outer: int =
                                    common.OUTER_FEELER_PERIPHERAL_BATCH_SIZE,
            num_peripheral_batches_outer: int =
                                    common.OUTER_FEELER_PERIPHERAL_BATCHES,
            regularize_jitter_multiply: float = 0.,
            regularize_jitter_add: float = 0.,
            **network_setup_args,
    ) -> None:

        if self._skip_when_profile(profile):
            return

        self.known_param_indices = known_param_indices
        self.inner_data_generator = _SamplingFeelerGenerator(
            estimates_min_and_max,
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
            peripheral_batch_size_inner,
            num_peripheral_batches_inner,
            regularize_jitter_multiply,
            regularize_jitter_add,
        )

        self.inner_feeler_net = _SamplingFeelerNet(
            self.inner_data_generator,
            num_unknown_param,
            num_known_param,
            profile,
            include_threshold=common.INNER_FEELER_INCLUDE_THRESHOLD,
            **network_setup_args,
        )

        self.inner_sampling_net = _ParamSamplingNet(
            self.inner_feeler_net,
            preprocess_params_fn,
            num_unknown_param,
            num_known_param,
            profile,
            **network_setup_args,
        )

        self.hits_inside_net = _IsInsideNet(
            sampling_distribution_fn,
            self.inner_sampling_net,
            preprocess_params_fn,
            num_stat,
            num_unknown_param,
            num_known_param,
            known_param_indices,
            profile,
            **network_setup_args,
        )

        self.outer_data_generator = _OuterFeelerGenerator(
            self.inner_sampling_net.sample_params,
            sampling_distribution_fn,
            preprocess_params_fn,
            self.is_inside,
            params_is_valid_fn,
            estimates_fn,
            num_unknown_param,
            num_known_param,
            profile,
            sample_size,
            sd_known,
            num_chains,
            chain_length,
            peripheral_batch_size_outer,
            num_peripheral_batches_outer,
            regularize_jitter_multiply,
            regularize_jitter_add,
        )

        self.outer_feeler_net = _SamplingFeelerNet(
            self.outer_data_generator,
            num_unknown_param,
            num_known_param,
            profile,
            include_threshold=common.OUTER_FEELER_INCLUDE_THRESHOLD,
            include_boost=common.OUTER_FEELER_INCLUDE_BOOST,
            **network_setup_args,
        )

        self.outer_sampling_net = _ParamSamplingNet(
            self.outer_feeler_net,
            preprocess_params_fn,
            num_unknown_param,
            num_known_param,
            profile,
            **network_setup_args,
        )

        super().__init__(
            subobjects_to_save={
                "innergen": self.inner_data_generator,
                "innerfeeler": self.inner_feeler_net,
                "innersampler": self.inner_sampling_net,
                "innerz": self.hits_inside_net,
                "outergen": self.outer_data_generator,
                "outerfeeler": self.outer_feeler_net,
                "outersampler": self.outer_sampling_net,
            },
        )

    @tf.function
    def is_inside(
            self,
            estimates: Tensor2[tf32, Samples, Stats],
            params: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[ttf.bool, Samples]:

        known_params = tf.gather(params, self.known_param_indices, axis=1)
        inside_net = self.hits_inside_net
        bools = inside_net.is_inside_sampled_region(estimates, known_params)
        return bools

    def fit(self, *args, **kwargs):
        self.inner_data_generator.fit()
        self.inner_feeler_net.fit(*args, **kwargs)
        self.inner_data_generator.release_gpu_memory()
        self.inner_sampling_net.fit(*args, **kwargs)
        self.hits_inside_net.fit(*args, **kwargs)

        self.outer_data_generator.fit()
        self.outer_feeler_net.fit(*args, **kwargs)
        self.outer_data_generator.release_gpu_memory()

        self.outer_sampling_net.fit(*args, **kwargs)

    @tf.function
    def sample_params(
            self,
            n_inner: int,
            n_outer: int = 0,
            known_mins_inner: Union[float,
                                    Tensor2[tf32,
                                            Samples,
                                            KnownParams]] = common.PARAMS_MIN,
            known_maxs_inner: Union[float,
                                    Tensor2[tf32,
                                            Samples,
                                            KnownParams]] = common.PARAMS_MAX,
            known_mins_outer: Union[float,
                                    Tensor2[tf32,
                                            Samples,
                                            KnownParams]] = common.PARAMS_MIN,
            known_maxs_outer: Union[float,
                                    Tensor2[tf32,
                                            Samples,
                                            KnownParams]] = common.PARAMS_MAX,
    ) -> Tensor2[tf32, Samples, Params]:

        params_inner = self.inner_sampling_net.sample_params(
            n_inner,
            known_min_vals=known_mins_inner,
            known_max_vals=known_maxs_inner
        )
        params_outer = self.outer_sampling_net.sample_params(
            n_outer,
            known_min_vals=known_mins_outer,
            known_max_vals=known_maxs_outer
        )
        params = tf.concat([params_inner, params_outer], axis=0)
        return params
