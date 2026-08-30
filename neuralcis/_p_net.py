import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
from ._param_sampler import _ParamSampler
from ._z_net import _ZNet
from ._data_saver import _DataSaver
from ._utils import known_params_from_params

# typing
from typing import Callable, Tuple
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
            interest_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor1[tf32, Samples]
            ],
            canonicalize_fn: Callable[
                [Tensor2[tf32, Samples, Stats],
                 Tensor2[tf32, Samples, Params],
                 Tensor1[tf32, Samples],
                 bool],
                Tuple[Tensor2[tf32, Samples, Stats],
                      Tensor2[tf32, Samples, Params],
                      Tensor1[tf32, Samples]],
            ],
            num_stat: int,
            num_unknown_param: int,
            num_known_param: int,
            num_stats_remaining_after_canonicalization: int,
            param_sampler: _ParamSampler,
            profile: str,
            **network_setup_args,
    ) -> None:

        if self._skip_when_profile(profile):
            return

        self.sampling_distribution_fn = sampling_distribution_fn
        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param

        self.param_sampler = param_sampler
        self.znet = _ZNet(
            self.sampling_distribution_fn,                                     # type: ignore
            self.param_sampler.sample_params,
            interest_fn,
            canonicalize_fn,
            num_stat,
            num_unknown_param,
            num_known_param,
            num_stats_remaining_after_canonicalization,
            profile,
            **network_setup_args,
        )

        super().__init__(
            subobjects_to_save={"znet": self.znet},
        )

    def fit(self, *args, **kwargs) -> None:
        return self.znet.fit(*args, **kwargs)

    def compile(self, *args, **kwargs) -> None:
        self.znet.compile(*args, **kwargs)

    def p(
            self,
            estimates: Tensor2[tf32, Samples, Stats],
            params_null: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        z = self.znet.z(estimates, params_null)
        return self.p_from_z(z)

    @tf.function
    def p_from_interest(
            self,
            estimates: Tensor2[tf32, Samples, Stats],
            interest: Tensor1[tf32, Samples],
            known_params: Tensor2[tf32, Samples, KnownParams],
    ) -> Tensor1[tf32, Samples]:

        # TODO: This should probably be the main p function and the other one
        #       could be done away with.  But will need to reformulate the
        #       users of this func.

        z = self.znet.call_tf_interest_only(estimates, interest, known_params)
        return self.p_from_z(z[:, None])

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
            profile: str,
    ):

        values = {}

        if self.znet.is_available_in_profile(profile):
            zs = self.znet.call_tf((estimates, params_null))
            ps = self.p_from_z(zs[:, 0])
            for i in range(zs.shape[-1]):
                values[f"z{i}"] = zs[:, i]
            values["p"] = ps

        if self.param_sampler.is_available_in_profile(profile):
            inner_feeler = self.param_sampler.inner_feeler_net
            if inner_feeler.is_available_in_profile(profile):
                inner_feeler_outputs = inner_feeler.call_tf((params_null,
                                                             params_null))     # type: ignore
                inner_importance = inner_feeler.get_log_importance_from_net(
                    params_null,
                )
                values["inner_log_vol"] = inner_feeler_outputs[:, 0]
                values["inner_include"] = inner_feeler_outputs[:, 1]
                values["inner_importance"] = inner_importance

            outer_feeler = self.param_sampler.outer_feeler_net
            if outer_feeler.is_available_in_profile(profile):
                outer_feeler_outputs = outer_feeler.call_tf((params_null,
                                                             params_null))     # type: ignore
                outer_importance = outer_feeler.get_log_importance_from_net(
                    params_null
                )
                values["outer_log_vol"] = outer_feeler_outputs[:, 0]
                values["outer_include"] = outer_feeler_outputs[:, 1]
                values["outer_importance"] = outer_importance

            inside_net = self.param_sampler.hits_inside_net
            if inside_net.is_available_in_profile(profile):
                known_params_null = known_params_from_params(
                    params_null,
                    self.num_unknown_param,
                    self.num_known_param,
                )
                inside_prob = inside_net.call_tf((
                    estimates,
                    known_params_null,
                ))
                hits_inside = inside_net.is_inside_sampled_region(
                    estimates,
                    known_params_null,
                )
                values["inside_prob"] = inside_prob
                values["hits_inside"] = hits_inside

        return values

    def num_param(self) -> int:
        return self.num_unknown_param + self.num_known_param
