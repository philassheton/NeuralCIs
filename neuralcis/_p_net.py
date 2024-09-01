import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
from neuralcis._z_net import _ZNet
from neuralcis._sampling_feeler_net import _SamplingFeelerNet
from neuralcis._param_sampling_net import _ParamSamplingNet
from neuralcis._data_saver import _DataSaver
from neuralcis import common

# typing
from typing import Callable, Tuple
from tensor_annotations.tensorflow import Tensor1, Tensor2
from tensor_annotations.tensorflow import float32 as tf32
from neuralcis.common import Samples, Params, Estimates, Zs, MinAndMax

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
            estimates_min_and_max: Tensor2[tf32, Estimates, MinAndMax],
            num_unknown_param: int,
            num_known_param: int,
            **network_setup_args,
    ) -> None:
        self.sampling_distribution_fn = sampling_distribution_fn
        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param

        known_param_indices = [
            i + num_unknown_param for i in range(num_known_param)
        ]
        # TODO: These should live a layer higher, so they can also be used
        #       by the CINet.  Will need a wrapper object that includes these
        #       and p-net and CInet.  (Doesn't belong in NeuralCIs object).
        self.feeler_net = _SamplingFeelerNet(
            estimates_min_and_max,
            self.sampling_distribution_fn,
            num_unknown_param,
            num_known_param,
            **network_setup_args,
        )
        self.param_sampling_net = _ParamSamplingNet(
            num_unknown_param + num_known_param,
            self.sampling_distribution_fn,                                     # type: ignore
            self.feeler_net,
            **network_setup_args,
        )

        self.validation_params = self.sample_params(common.VALIDATION_SET_SIZE)
        self.validation_estimates = self.sampling_distribution_fn(
            self.validation_params
        )

        self.znet = _ZNet(
            self.sampling_distribution_fn,                                     # type: ignore
            self.sample_params,
            contrast_fn,
            self.validation_set,                                               # type: ignore
            known_param_indices,
            **network_setup_args,
        )

        super().__init__(
            subobjects_to_save={"znet": self.znet}
        )

    def fit(self, *args, **kwargs) -> None:
        self.feeler_net.fit(*args, **kwargs)
        self.param_sampling_net.fit(*args, **kwargs)
        self.znet.fit(*args, **kwargs)

    def compile(self, *args, **kwargs) -> None:
        self.feeler_net.compile(*args, **kwargs)
        self.param_sampling_net.compile(*args, **kwargs)
        self.znet.compile(*args, **kwargs)

    @tf.function
    def sample_params(
            self,
            n: int,
    ) -> Tensor2[tf32, Samples, Params]:

        us = tf.random.uniform((n, self.num_param()),
                               minval=tf.constant(common.PARAMS_MIN),
                               maxval=tf.constant(common.PARAMS_MAX))
        params = self.param_sampling_net.call_tf(us)
        return params                                                          # type: ignore

    @tf.function
    def validation_set(
            self,
    ) -> Tuple[
        Tensor2[tf32, Samples, Estimates],
        Tensor2[tf32, Samples, Params],
    ]:

        return self.validation_estimates, self.validation_params

    @tf.function
    def p(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            params_null: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        zs = self.znet.call_tf((estimates, params_null))
        return self.ps_from_zs(zs)                                             # type: ignore

    @tf.function
    def ps_from_zs(
            self,
            zs: Tensor2[tf32, Samples, Zs],
    ):

        cdf = tfp.distributions.Normal(0., 1.).cdf(zs[:, 0])
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

        zs = self.znet.call_tf((estimates, params_null))
        ps = self.ps_from_zs(zs)                                               # type: ignore
        feeler_outputs = self.feeler_net.call_tf(params_null)

        values = {}
        for i in range(zs.shape[-1]):
            values[f"z{i}"] = zs[:, i]
        values["p"] = ps
        values["feeler_log_vol"] = feeler_outputs[:, 0]
        values["feeler_p_intersect"] = feeler_outputs[:, 1]

        return values

    @tf.function
    def num_param(self) -> int:
        return self.num_unknown_param + self.num_known_param
