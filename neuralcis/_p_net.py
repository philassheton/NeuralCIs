import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
from neuralcis._param_sampling_net import _ParamSamplingNet
from neuralcis._z_net import _ZNet
from neuralcis._data_saver import _DataSaver
from neuralcis import common

# typing
from typing import Callable, Tuple, Sequence
from tensor_annotations.tensorflow import Tensor1, Tensor2
from tensor_annotations.tensorflow import float32 as tf32
from neuralcis.common import Samples, Estimates, Params, KnownParams, Zs

NetInputBlob = Tuple[
    Tensor2[tf32, Samples, Estimates],
    Tensor2[tf32, Samples, Params],
]


class _PNet(_DataSaver):
    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor2[tf32, Samples, Estimates]
            ],
            contrast_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor1[tf32, Samples]
            ],
            transform_on_params_fn: Callable[
                [Tensor2[tf32, Samples, Estimates],
                 Tensor2[tf32, Samples, Params]],
                Tuple[Tensor2[tf32, Samples, Estimates],
                      Tensor2[tf32, Samples, Params]]
            ],
            num_unknown_param: int,
            num_known_param: int,
            known_param_indices: Sequence[int],
            num_params_remaining_after_transform: int,
            param_sampling_net: _ParamSamplingNet,
            **network_setup_args,
    ) -> None:
        self.sampling_distribution_fn = sampling_distribution_fn
        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param

        self.param_sampling_net = param_sampling_net
        self.znet = _ZNet(
            self.sampling_distribution_fn,                                     # type: ignore
            self.param_sampling_net.sample_params,
            contrast_fn,
            transform_on_params_fn,
            num_unknown_param,
            num_known_param,
            known_param_indices,
            num_params_remaining_after_transform,
            **network_setup_args,
        )

        super().__init__(
            subobjects_to_save={"znet": self.znet}
        )

    def fit(self, *args, **kwargs) -> None:
        self.znet.fit(*args, **kwargs)

    def compile(self, *args, **kwargs) -> None:
        self.znet.compile(*args, **kwargs)

    @tf.function
    def p(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            params_null: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        z = self.znet.z(estimates, params_null)
        return self.p_from_z(z)

    @tf.function
    def p_from_contrast(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
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
            estimates: Tensor2[tf32, Samples, Estimates],
            params_null: Tensor2[tf32, Samples, Params],
    ):

        # TODO: Check this; have kept separate from p() rather than refactoring
        #       as I *think* the graph will be more efficient for p() when
        #       intermediate results are not maintained, and since p() is
        #       used in training of CINet, this is important.  Not sure though
        #       and should check.

        zs = self.znet.call_tf_transformed((estimates, params_null))
        ps = self.p_from_z(zs[:, 0])
        feeler_net = self.param_sampling_net.feeler_net
        feeler_outputs = feeler_net.call_tf(params_null)                       # type: ignore
        feeler_final = feeler_net.get_log_importance_from_net(params_null)

        values = {}
        for i in range(zs.shape[-1]):
            values[f"z{i}"] = zs[:, i]
        values["p"] = ps
        values["feeler_log_vol"] = feeler_outputs[:, 0]
        values["feeler_p_intersect"] = feeler_outputs[:, 1]
        values["feeler"] = feeler_final

        return values

    @tf.function
    def num_param(self) -> int:
        return self.num_unknown_param + self.num_known_param
