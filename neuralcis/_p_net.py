import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
from neuralcis._z_net import _ZNet
from neuralcis._coordi_net import _CoordiNet
from neuralcis._estimator_net import _EstimatorNet
from neuralcis._data_saver import _DataSaver
from neuralcis import common

# typing
from typing import Callable, Tuple
from tensor_annotations.tensorflow import Tensor1, Tensor2
from tensor_annotations.tensorflow import float32 as tf32
from neuralcis.common import Samples, Params, Estimates

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
            num_unknown_param: int,
            num_known_param: int,
            filename: str = "",
            **network_setup_args,
    ) -> None:
        self.sampling_distribution_fn = sampling_distribution_fn
        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param

        self.validation_params = self.sample_params(common.VALIDATION_SET_SIZE)
        self.validation_estimates = self.sampling_distribution_fn(
            self.validation_params
        )

        known_param_indices = [
            i + num_unknown_param for i in range(num_known_param)
        ]
        self.estimatornet = _EstimatorNet(
            sampling_distribution_fn,
            self.sample_params,
            contrast_fn,
            **network_setup_args,
        )
        self.coordinet = _CoordiNet(
            self.estimatornet,
            sampling_distribution_fn,
            self.sample_params,
            **network_setup_args,
        )
        self.znet = _ZNet(
            self.sampling_distribution_fn,                                     # type: ignore
            self.sample_params,
            contrast_fn,
            self.validation_set,
            known_param_indices,
            self.coordinet.call_tf,                                            # type: ignore
            **network_setup_args,
        )

        super().__init__(
            filename=filename,
            subobjects_to_save={"znet": self.znet,
                                "estimatornet": self.estimatornet,
                                "coordinet": self.coordinet}
        )

    def fit(self, *args, **kwargs) -> None:
        self.estimatornet.fit(*args, **kwargs)
        self.coordinet.fit(*args, **kwargs)
        self.znet.fit(*args, **kwargs)

    def compile(self, *args, **kwargs) -> None:
        self.estimatornet.compile(*args, **kwargs)
        self.coordinet.compile(*args, **kwargs)
        self.znet.compile(*args, **kwargs)


    @tf.function
    def sample_params(
            self,
            n: int,
    ) -> Tensor2[tf32, Samples, Params]:

        return tf.random.uniform(
            (n, self.num_param()),
            minval=tf.constant(-1.),
            maxval=tf.constant(1.),
        )

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
        cdf = tfp.distributions.Normal(0., 1.).cdf(zs[:, 0])

        p: Tensor1[tf32, Samples] = 1. - tf.math.abs(cdf*2. - 1.)              # type: ignore

        return p

    @tf.function
    def num_param(self) -> int:
        return self.num_unknown_param + self.num_known_param
