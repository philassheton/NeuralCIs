from neuralcis._simulator_net import _SimulatorNet
from neuralcis._sampling_feeler_net import _SamplingFeelerNet
from neuralcis import _utils
from neuralcis import common
import tensorflow as tf

# Typing
from typing import Tuple, Callable, Optional
from neuralcis.common import Samples, Params, Us, Estimates, MinAndMax
from neuralcis.common import NetTargetBlob, NetInputs
from tensor_annotations import tensorflow as ttf
from tensor_annotations.tensorflow import Tensor1, Tensor2
tf32 = ttf.float32

NetInputBlob = Tensor2[tf32, Samples, Us]
NetOutputBlob = Tuple[Tensor2[tf32, Samples, Params],  # net outputs (params)
                      Tensor1[tf32, Samples]]          # Jacobian determinants


class _ParamSamplingNet(_SimulatorNet):
    def __init__(
            self,
            num_unknown_param: int,
            num_known_param: int,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],                 # params
                Tensor2[tf32, Samples, Estimates],                # -> ys
            ],
            estimates_min_and_max: Tensor2[tf32, Estimates, MinAndMax],
            **network_setup_args,
    ) -> None:

        # TODO: Refactor so this isn't necessary
        tf.keras.models.Model.__init__(self)

        self.feeler_net = _SamplingFeelerNet(
            estimates_min_and_max,
            sampling_distribution_fn,
            num_unknown_param,
            num_known_param,
            **network_setup_args,
        )

        self.num_param = num_unknown_param + num_known_param
        self.sampling_distribution_fn = sampling_distribution_fn

        super().__init__(
            num_outputs_for_each_net=(self.num_param,),
            subobjects_to_save={'feelernet': self.feeler_net},
            **network_setup_args,
        )

    def simulate_training_data(
            self,
            n: ttf.int32,
    ) -> Tuple[NetInputBlob, None]:

        us = tf.random.uniform((n, self.num_param),
                               minval=common.PARAMS_MIN,
                               maxval=common.PARAMS_MAX)
        return us, None

    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: Optional[NetTargetBlob] = None,
    ) -> ttf.float32:

        eps = common.SMALLEST_LOGABLE_NUMBER
        params, jacobdets = net_outputs

        importance_log = self.feeler_net.get_log_importance_from_net(params)

        jacobdets_floored = _utils._soft_floor_at_zero(jacobdets)
        jacobdets_log = tf.math.log(jacobdets_floored + eps)

        neg_log_likelihoods = -importance_log - jacobdets_log

        return tf.math.reduce_mean(neg_log_likelihoods)

    def net_inputs(
            self,
            inputs: NetInputBlob
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], ...]:

        # TODO: This should be pushed up to the _SimulatorNet I think.
        return [inputs]

    def call_tf_training(
            self,
            input_blob: NetInputBlob
            ) -> NetOutputBlob:

        us = input_blob

        with tf.GradientTape() as tape:  # type: ignore
            tape.watch(us)
            net_inputs = self.net_inputs(us)
            params = self._call_tf(net_inputs, training=True)

        jacobians = tape.batch_jacobian(params, us)
        jacobdets = tf.linalg.det(jacobians)

        return params, jacobdets                                               # type: ignore

    def fit(self, *args, **kwargs):
        self.feeler_net.fit(*args, **kwargs)
        super().fit(*args, **kwargs)

    def compile(self, *args, **kwargs):
        self.feeler_net.compile(*args, **kwargs)
        super().compile(*args, **kwargs)

    @tf.function
    def sample_params(
            self,
            n: int,
    ) -> Tensor2[tf32, Samples, Params]:

        us = tf.random.uniform((n, self.num_param),
                               minval=tf.constant(common.PARAMS_MIN),
                               maxval=tf.constant(common.PARAMS_MAX))
        params = self.call_tf(us)
        return params                                                          # type: ignore
