import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
from neuralcis._z_net import _ZNet
from neuralcis._simulator_net import _SimulatorNet
from neuralcis import common

# typing
from typing import Callable, Tuple
from tensor_annotations.tensorflow import Tensor1, Tensor2
from tensor_annotations.tensorflow import float32 as tf32
from neuralcis.common import Samples, Params, Estimates, Ys, NetInputs
from neuralcis.common import FocalParam, NuisanceParams, KnownParams
import tensor_annotations.tensorflow as ttf

NetOutputBlob = Tensor2[tf32, Samples, NuisanceParams]
NetTargetBlob = Tuple[Tensor2[tf32, Samples, Estimates],
                      Tensor2[tf32, Samples, Params]]


# TODO: At present this just provides a p-value wrt the first param, with the
#       other unknown params maximised over to remove them (crude treatment of
#       nuisance variables).  This will be expanded to provide a more flexible
#       and less conservative treatment of nuisance variables.

class _PNet(_SimulatorNet):
    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor2[tf32, Samples, Estimates]
            ],
            num_unknown_param: int,
            num_known_param: int,
            filename: str = "",
            **network_setup_args
    ) -> None:

        # TODO: rewrite this so that the _SimulatorNet is in a separate object
        #       so that cases with no nuisance variables don't have to train
        #       an empty network
        self.sampling_distribution_fn = sampling_distribution_fn
        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param

        self.validation_params = self.sample_params(common.VALIDATION_SET_SIZE)
        self.validation_estimates = self.sampling_distribution(
            self.validation_params
        )

        num_estimate = int(tf.shape(self.validation_estimates)[1])
        self.chisq_distribution = tfp.distributions.Chi2(num_estimate)

        self.znet = _ZNet(
            self.sampling_distribution,                                        # type: ignore
            self.sample_params,
            self.validation_set                                                # type: ignore
        )

        super().__init__(
            num_outputs=num_unknown_param - 1,
            filename=filename,
            subobjects_to_save={"znet": self.znet},
            **network_setup_args
        )

    ###########################################################################
    #
    #   _SimulatorNet implementation
    #
    ###########################################################################

    @tf.function
    def simulate_training_data(
            self,
            n: ttf.int32
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], NetTargetBlob]:

        estimates, params = self.sample_estimates_and_params(n)
        target_blob = (estimates, params)
        net_inputs = self.net_inputs(*target_blob)

        return net_inputs, target_blob

    @tf.function
    def get_validation_set(
            self
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], NetTargetBlob]:

        target_blob = (self.validation_estimates, self.validation_params)
        net_inputs = self.net_inputs(*target_blob)

        return net_inputs, target_blob

    @tf.function
    def loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: NetTargetBlob
    ) -> ttf.float32:

        estimates, params = target_outputs
        params_denuisanced = self.insert_nuisance_params(params, net_outputs)
        p = self.p_simultaneous(estimates, params_denuisanced)

        return -tf.reduce_sum(p)                                               # type: ignore

    ###########################################################################
    #
    #  Other
    #
    ###########################################################################

    def fit(self, *args, **kwargs) -> None:
        self.znet.fit(*args, **kwargs)
        super().fit(*args, **kwargs)

    @tf.function
    def sampling_distribution(
            self,
            params: Tensor2[tf32, Samples, Params],
    ) -> Tensor2[tf32, Samples, Estimates]:

        return self.sampling_distribution_fn(params)

    @tf.function
    def sample_estimates_and_params(
            self,
            n: ttf.int32
    ) -> Tuple[Tensor2[tf32, Samples, Estimates],
               Tensor2[tf32, Samples, Params]]:

        params = self.sample_params(n)
        estimates = self.sampling_distribution(params)

        return estimates, params

    @tf.function
    def sample_params(
            self,
            n: ttf.int32
    ) -> Tensor2[tf32, Samples, Params]:

        return tf.random.uniform(
            (n, self.num_param()),
            minval=tf.constant(-1.),
            maxval=tf.constant(1.)
        )

    @tf.function
    def split_params(
            self,
            params: Tensor2[tf32, Samples, Params]
    ) -> Tuple[
        Tensor2[tf32, Samples, FocalParam],
        Tensor2[tf32, Samples, NuisanceParams],
        Tensor2[tf32, Samples, KnownParams]
    ]:

        focal_param = params[:, 0:1]
        nuisance_params = params[:, 1:self.num_unknown_param]
        known_params = params[:, self.num_unknown_param:]

        return focal_param, nuisance_params, known_params                      # type: ignore

    @tf.function
    def validation_set(
            self
    ) -> Tuple[
        Tensor2[tf32, Samples, Estimates],
        Tensor2[tf32, Samples, Params]
    ]:

        return self.validation_estimates, self.validation_params

    @tf.function
    def p(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            params_null: Tensor2[tf32, Samples, Params]
    ) -> Tensor1[tf32, Samples]:

        # TODO: This is replicating some functionality from loss: REFACTOR
        # TODO: Also would be cleaner if this accepted only focal_param and
        #       known_params since it will anyway ignore nuisance_params
        net_inputs = self.net_inputs(estimates, params_null)
        worst_nuisance_params = self.net(net_inputs)
        params = self.insert_nuisance_params(
            params_null,
            worst_nuisance_params
        )
        return self.p_simultaneous(estimates, params)

    @tf.function
    def p_simultaneous(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            params: Tensor2[tf32, Samples, Params]
    ) -> Tensor1[tf32, Samples]:

        y: Tensor2[tf32, Samples, Ys] = estimates                              # type:ignore
        z = self.znet.call_tf(y, params)
        chisq = tf.reduce_sum(tf.math.square(z), axis=1)
        cdf = self.chisq_distribution.cdf(chisq)
        p = 1. - cdf

        return p

    def net_inputs(
            self,
            estimates: Tensor2[tf32, Samples, Estimates],
            params: Tensor2[tf32, Samples, Params]
    ) -> Tensor2[tf32, Samples, NetInputs]:

        focal_param, nuisance_params, known_params = self.split_params(params)
        return tf.concat([estimates, focal_param, nuisance_params], axis=1)

    def insert_nuisance_params(
            self,
            params: Tensor2[tf32, Samples, Params],
            new_nuisance_params: Tensor2[tf32, Samples, NuisanceParams]
    ) -> Tensor2[tf32, Samples, Params]:

        focal_param, nuisance_params, known_params = self.split_params(params)
        new_params = tf.concat(
            [focal_param, new_nuisance_params, known_params],
            axis=1
        )
        return new_params

    def num_param(self) -> int:
        return self.num_unknown_param + self.num_known_param
