import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
from ._param_sampler import _ParamSampler
from ._z_net import _ZNet
from ._data_saver import _DataSaver

# typing
from typing import Callable, Tuple, Sequence
from tensor_annotations.tensorflow import Tensor1, Tensor2
from tensor_annotations.tensorflow import float32 as tf32
from .common import Samples, Stats, Params, KnownParams

NetInputBlob = Tuple[
    Tensor2[tf32, Samples, Stats],
    Tensor2[tf32, Samples, Params],
]


class _PNet(_DataSaver):
    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor2[tf32, Samples, Stats]
            ],
            contrast_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor1[tf32, Samples]
            ],
            transform_on_params_fn: Callable[
                [Tensor2[tf32, Samples, Stats],
                 Tensor2[tf32, Samples, Params]],
                Tuple[Tensor2[tf32, Samples, Stats],
                      Tensor2[tf32, Samples, Params]]
            ],
            transform_on_stats_fn: Callable[
                [Tensor2[tf32, Samples, Stats],
                 Tensor2[tf32, Samples, Params]],
                Tuple[Tensor2[tf32, Samples, Stats],
                      Tensor2[tf32, Samples, Params]],
            ],
            num_stat: int,
            num_unknown_param: int,
            num_known_param: int,
            known_param_indices: Sequence[int],
            num_params_remaining_after_transform: int,
            num_stats_remaining_after_transform: int,
            param_sampler: _ParamSampler,
            profile: str,
            **network_setup_args,
    ) -> None:

        if self._skip_when_profile(profile):
            return

        self.sampling_distribution_fn = sampling_distribution_fn
        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param
        self.known_param_indices = known_param_indices

        self.param_sampler = param_sampler
        self.znet = _ZNet(
            self.sampling_distribution_fn,                                     # type: ignore
            self.param_sampler.sample_params,
            contrast_fn,
            transform_on_params_fn,
            transform_on_stats_fn,
            num_stat,
            num_unknown_param,
            num_known_param,
            known_param_indices,
            num_params_remaining_after_transform,
            num_stats_remaining_after_transform,
            profile,
            **network_setup_args,
        )

        super().__init__(
            subobjects_to_save={"znet": self.znet},
        )

    def fit(self, *args, **kwargs) -> None:
        self.znet.fit(*args, **kwargs)

    def compile(self, *args, **kwargs) -> None:
        self.znet.compile(*args, **kwargs)

    @tf.function
    def p(
            self,
            estimates: Tensor2[tf32, Samples, Stats],
            params_null: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        z = self.znet.z(estimates, params_null)
        return self.p_from_z(z)

    @tf.function
    def p_from_contrast(
            self,
            estimates: Tensor2[tf32, Samples, Stats],
            contrast: Tensor1[tf32, Samples],
            known_params: Tensor2[tf32, Samples, KnownParams],
    ) -> Tensor1[tf32, Samples]:

        # TODO: This should probably be the main p function and the other one
        #       could be done away with.  But will need to reformulate the
        #       users of this func.

        z = self.znet.call_tf_contrast_only(estimates, contrast, known_params)
        return self.p_from_z(z[:, None])

    @tf.function
    def p_from_z(
            self,
            z: Tensor1[tf32, Samples],
    ) -> Tensor1[tf32, Samples]:

        cdf = tfp.distributions.Normal(0., 1.).cdf(z)
        p = 1. - tf.math.abs(cdf * 2. - 1.)

        return p                                                               # type: ignore

    @tf.function
    def p_workings(
            self,
            estimates: Tensor2[tf32, Samples, Stats],
            params_null: Tensor2[tf32, Samples, Params],
    ):

        # TODO: Check this; have kept separate from p() rather than refactoring
        #       as I *think* the graph will be more efficient for p() when
        #       intermediate results are not maintained, and since p() is
        #       used in training of CINet, this is important.  Not sure though
        #       and should check.

        zs = self.znet.call_tf_transformed((estimates, params_null))
        ps = self.p_from_z(zs[:, 0])
        inner_feeler = self.param_sampler.inner_feeler_net
        inner_feeler_outputs = inner_feeler.call_tf((params_null, params_null))  # type: ignore
        inner_importance = inner_feeler.get_log_importance_from_net(
            params_null
        )
        outer_feeler = self.param_sampler.outer_feeler_net
        outer_feeler_outputs = outer_feeler.call_tf((params_null, params_null))  # type: ignore
        outer_importance = outer_feeler.get_log_importance_from_net(
            params_null
        )

        known_params_null = tf.gather(params_null,
                                      self.known_param_indices,
                                      axis=1)
        inside_net = self.param_sampler.hits_inside_net
        inside_prob = inside_net.call_tf((estimates, known_params_null))
        hits_inside = inside_net.is_inside_sampled_region(estimates,
                                                          known_params_null)

        values = {}
        for i in range(zs.shape[-1]):
            values[f"z{i}"] = zs[:, i]
        values["p"] = ps
        values["inner_log_vol"] = inner_feeler_outputs[:, 0]
        values["inner_include"] = inner_feeler_outputs[:, 1]
        values["inner_importance"] = inner_importance
        values["outer_log_vol"] = outer_feeler_outputs[:, 0]
        values["outer_include"] = outer_feeler_outputs[:, 1]
        values["outer_importance"] = outer_importance
        values["inside_prob"] = inside_prob
        values["hits_inside"] = hits_inside

        return values

    @tf.function
    def num_param(self) -> int:
        return self.num_unknown_param + self.num_known_param
