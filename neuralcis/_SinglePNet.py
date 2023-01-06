import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
from neuralcis._SingleZNet import _SingleZNet
from neuralcis import common

# typing
from typing import Callable, Tuple
from tensor_annotations.tensorflow import Tensor1, Tensor2
from tensor_annotations.tensorflow import float32 as tf32
from neuralcis.common import Samples, Params, Estimates
import tensor_annotations.tensorflow as ttf


class _SinglePNet:
    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor2[tf32, Samples, Estimates]
            ],
            num_known_params: int,
            two_sided: bool = True
    ) -> None:

        self.sampling_distribution_fn = sampling_distribution_fn
        self.num_param = num_known_params + 1
        self.two_sided = tf.constant(two_sided)
        self.z_distribution = tfp.distributions.Normal(0., 1.)

        self.validation_params = self.sample_params(common.VALIDATION_SET_SIZE)
        self.validation_estimates = self.sampling_distribution(
            self.validation_params
        )

        self.znet = _SingleZNet(
            self.sampling_distribution,
            self.sample_params,
            self.validation_set
        )

    def fit(self, *args, **kwargs) -> None:
        self.znet.fit(*args, **kwargs)

    @tf.function
    def sampling_distribution(
            self,
            params: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        return self.sampling_distribution_fn(params)[:, 0]                     # type: ignore

    @tf.function
    def sample_params(
            self,
            n: ttf.int32
    ) -> Tensor2[tf32, Samples, Params]:

        return tf.random.uniform(
            (n, self.num_param),
            minval=tf.constant(-1.),
            maxval=tf.constant(1.)
        )

    @tf.function
    def validation_set(
            self
    ) -> Tuple[Tensor1[tf32, Samples], Tensor2[tf32, Samples, Params]]:

        # TODO: now that y is encoded as "estimates", it feels like it
        #       might be more logical to code _SingleZNet to always have
        #       params followed by y, to follow the convention of
        #       (params, estimates) pairs in the rest of the code (or
        #       even better reverse (params, estimates) so that we get
        #       (estimates | params)).
        return self.validation_estimates, self.validation_params

    @tf.function
    def p(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            params: Tensor2[tf32, Samples, Params]
    ) -> Tensor1[tf32, Samples]:

        z = self.znet.call_tf(estimates[:, 0], params)                         # type: ignore
        cdf = self.z_distribution.cdf(z)
        p = tf.cond(
            self.two_sided,
            lambda: 1. - tf.math.abs(1. - 2.*cdf),
            lambda: cdf
        )
        return p
