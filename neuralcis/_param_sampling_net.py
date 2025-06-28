from ._simulator_net import _SimulatorNet
from ._sampling_feeler_net import _SamplingFeelerNet
from .common import TESTING
from . import _utils, common
import tensorflow as tf

# Typing
from typing import Tuple, Callable, Optional, Union
from .common import Samples, Params, KnownParams, Zs, Us
from .common import NetTargetBlob, NetInputs, NetOutputs
from tensor_annotations import tensorflow as ttf
from tensor_annotations.tensorflow import Tensor1, Tensor2
tf32 = ttf.float32

NetInputBlob = Tuple[Tensor2[tf32, Samples, Zs],       # unknown param uniforms
                     Tensor2[tf32, Samples, Us]]       # known param uniforms
NetOutputBlob = Tuple[Tensor2[tf32, Samples, Params],  # net outputs (params)
                      Tensor1[tf32, Samples]]          # Jacobian determinants


class _ParamSamplingNet(_SimulatorNet):
    absolute_loss_increase_tol = common.ABS_LOSS_INCREASE_TOL_PARAM_SAMP_NET
    smallest_profile_found_in = TESTING

    def __init__(
            self,
            feeler_net: _SamplingFeelerNet,
            preprocess_params_fn: Callable[
                [Tensor2[tf32, Samples, Params], bool],
                Tensor2[tf32, Samples, Params]
            ],
            num_unknown_param: int,
            num_known_param: int,
            profile: str,
            **network_setup_args,
    ) -> None:

        super().__init__(
            profile,
            num_inputs_for_each_net=(num_unknown_param + num_known_param,),
            num_outputs_for_each_net=(num_unknown_param,),
            **network_setup_args,
        )

        if self._skip_when_profile(profile):
            return

        self.feeler_net = feeler_net
        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param
        self.preprocess_params_fn = preprocess_params_fn

    @tf.function
    def simulate_training_data(
            self,
    ) -> Tuple[NetInputBlob, None]:

        n = self.batch_size
        zs_us = self.simulate_zs_and_us(n)
        nothing = tf.zeros((n, 0))
        return zs_us, nothing

    @tf.function
    def simulate_zs_and_us(
            self,
            n: int,
            preprocess: bool = True,
            known_min_vals: Union[float, Tensor2[tf32, Samples, KnownParams]] =
                                                             common.PARAMS_MIN,
            known_max_vals: Union[float, Tensor2[tf32, Samples, KnownParams]] =
                                                             common.PARAMS_MAX,
    ) -> NetInputBlob:

        zs_unknown = tf.random.normal((n, self.num_unknown_param))
        us_known = tf.random.uniform((n, self.num_known_param),
                                     minval=known_min_vals,
                                     maxval=known_max_vals)
        if preprocess:
            us_known = self.preprocess_params_fn(us_known,
                                                 known_params_only=True)

        return zs_unknown, us_known

    @tf.function
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

    @tf.function
    def net_inputs(
            self,
            inputs: NetInputBlob
    ) -> Tuple[Tensor2[tf32, Samples, NetInputs], ...]:

        zs_unknown, us_known = inputs
        net_inputs = tf.concat([zs_unknown, us_known], axis=1)
        return (net_inputs,)

    @tf.function
    def call_tf(
            self,
            input_blob: NetInputBlob
    ) -> Tensor2[tf32, Samples, NetOutputs]:

        zs_unknown, us_known = input_blob
        net_inputs = self.net_inputs((zs_unknown, us_known))
        params_unknown = self._call_tf(net_inputs, training=False)
        params = tf.concat([params_unknown, us_known], axis=1)

        return params

    @tf.function
    def call_tf_training(
            self,
            input_blob: NetInputBlob
    ) -> NetOutputBlob:

        zs_unknown, us_known = input_blob

        with tf.GradientTape() as tape:  # type: ignore
            tape.watch(zs_unknown)
            net_inputs = self.net_inputs((zs_unknown, us_known))
            params_unknown = self._call_tf(net_inputs, training=True)

        jacobians = tape.batch_jacobian(params_unknown, zs_unknown)
        jacobdets = tf.linalg.det(jacobians)

        params = tf.concat([params_unknown, us_known], axis=1)

        return params, jacobdets                                               # type: ignore

    @tf.function
    def num_param(self) -> int:
        return self.num_unknown_param + self.num_known_param

    @tf.function
    def sample_params(
            self,
            n: int,
            preprocess: bool = True,
            known_min_vals: Union[float, Tensor2[tf32, Samples, KnownParams]] =
                                                             common.PARAMS_MIN,
            known_max_vals: Union[float, Tensor2[tf32, Samples, KnownParams]] =
                                                             common.PARAMS_MAX,
    ) -> Tensor2[tf32, Samples, Params]:

        zs_us = self.simulate_zs_and_us(n, preprocess,
                                        known_min_vals,
                                        known_max_vals)
        params = self.call_tf(zs_us)

        # TODO: A bit ugly, but we will have for now to preprocess twice.
        #       First, we preprocess the known params so that we get the
        #       other params coming out consistent with the preprocessed known
        #       params.  Then we use that to generate those other params and
        #       then need to preprocess to generate those.  Worth thinking
        #       if there is a cleaner way.  Preprocessing the known params
        #       twice feels a bit shaky.
        if preprocess:
            params = self.preprocess_params_fn(params)

        return params
