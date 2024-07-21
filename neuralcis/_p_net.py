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
from neuralcis.common import Samples, Params, Estimates, Ys
from neuralcis.common import NetInputs, NetOutputs

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
            **network_setup_args
    ) -> None:
        self.sampling_distribution_fn = sampling_distribution_fn
        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param

        self.validation_params = self.sample_params(common.VALIDATION_SET_SIZE)
        self.validation_estimates = self.sampling_distribution_fn(
            self.validation_params
        )

        self.estimatornet = _EstimatorNet(
            sampling_distribution_fn,
            self.sample_params,
            contrast_fn,
            **network_setup_args
        )
        self.coordinet = _CoordiNet(
            self.estimatornet,
            sampling_distribution_fn,
            self.sample_params,
            **network_setup_args
        )
        self.znet = _ZNet(
            self.sampling_distribution_remapped,                               # type: ignore
            self.sample_params,
            contrast_fn,
            self.validation_set_remapped,                                      # type: ignore
            [i + num_unknown_param for i in range(num_known_param)],
            **network_setup_args
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
    def sampling_distribution_remapped(
            self,
            params: Tensor2[tf32, Samples, Params],
    ) -> Tensor2[tf32, Samples, Estimates]:

        estimates = self.sampling_distribution_fn(params)
        inputs: Tensor2[tf32, Samples, NetInputs] = estimates                  # type: ignore
        outputs = self.coordinet.call_tf([inputs])
        estimate_coords: Tensor2[tf32, Samples, Estimates] = outputs           # type: ignore

        return estimate_coords

    @tf.function
    def sample_params(
            self,
            n: int,
    ) -> Tensor2[tf32, Samples, Params]:

        return tf.random.uniform(
            (n, self.num_param()),
            minval=tf.constant(-1.),
            maxval=tf.constant(1.)
        )

    @tf.function
    def validation_set(
            self
    ) -> Tuple[
        Tensor2[tf32, Samples, Estimates],
        Tensor2[tf32, Samples, Params]
    ]:

        return self.validation_estimates, self.validation_params

    @tf.function
    def validation_set_remapped(
            self
    ) -> Tuple[
        Tensor2[tf32, Samples, Estimates],
        Tensor2[tf32, Samples, Params]
    ]:

        inputs: Tensor2[tf32, Samples, NetInputs] = self.validation_estimates  # type: ignore
        outputs = self.coordinet.call_tf([inputs])
        estimates_remapped: Tensor2[tf32, Samples, Estimates] = outputs        # type: ignore

        return estimates_remapped, self.validation_params

    @tf.function
    def p(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            params_null: Tensor2[tf32, Samples, Params]
    ) -> Tensor1[tf32, Samples]:

        inputs: Tensor2[tf32, Samples, NetInputs] = estimates                  # type: ignore
        outputs = self.coordinet.call_tf([inputs])

        coords: Tensor2[tf32, Samples, Ys] = outputs                           # type: ignore
        zs = self.znet.call_tf_y_params(coords, params_null)

        z_focal: Tensor1[tf32, Samples] = zs[:, 0]                             # type: ignore
        cdf = tfp.distributions.Normal(0., 1.).cdf(z_focal)

        p: Tensor1[tf32, Samples] = 1. - tf.math.abs(cdf*2. - 1.)              # type: ignore

        return p

    @tf.function
    def num_param(self) -> int:
        return self.num_unknown_param + self.num_known_param
