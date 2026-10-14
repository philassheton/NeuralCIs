import tensorflow as tf
import tensorflow_probability as tfp

from ._p_net import _PNet
from ._simulator_net import _SimulatorNet
from ._utils import known_params_from_params
from . import common

# typing imports
from typing import Callable, Tuple, List, Sequence
from tensor_annotations.tensorflow import Tensor1, Tensor2
from .common import Samples, Params, Stats, NetInputs, NetOutputs, KnownParams
import tensor_annotations.tensorflow as ttf
tf32 = ttf.float32


NetInputBlob = Tuple[Tensor2[tf32, Samples, Stats],
                     Tensor2[tf32, Samples, KnownParams],
                     Tensor1[tf32, Samples]]                         # z-values
NetOutputBlob = Tensor2[tf32, Samples, NetOutputs]
NetTargetBlob = Tuple[Tensor2[tf32, Samples, Stats],
                      Tensor2[tf32, Samples, KnownParams],
                      Tensor1[tf32, Samples]]                        # z-values


class _CINet(_SimulatorNet):
    relative_loss_increase_tol = common.REL_LOSS_INCREASE_TOL_CI_NET
    def __init__(
            self,
            pnet: _PNet,               # TODO: make a protocol for PNets
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],
                Tensor2[tf32, Samples, Stats]
            ],
            sample_params_fn: Callable[
                [int],
                Tensor2[tf32, Samples, Params],
            ],
            canonicalize_fn: Callable[
                [Tensor2[tf32, Samples, Stats],
                 Tensor2[tf32, Samples, Params],
                 None,
                 bool],
                Tuple[Tensor2[tf32, Samples, Stats],
                      Tensor2[tf32, Samples, Params],
                      None],
            ],
            decanonicalize_interest_fn: Callable[
                [Tensor1[tf32, Samples], Tensor2[tf32, Samples, Stats]],
                Tensor1[tf32, Samples]
            ],
            num_estimate_remaining_after_canonicalization: int,
            num_unknown_param: int,
            num_known_param: int,
            profile: str,
            **network_setup_args,
    ) -> None:

        num_inputs = (num_estimate_remaining_after_canonicalization
                      + num_known_param + 1)                                   # estimates, known_params, target_z
        _SimulatorNet.__init__(self,
                               profile,
                               num_inputs_for_each_net=[num_inputs],
                               num_outputs_for_each_net=[1],
                               **network_setup_args)

        if self._skip_when_profile(profile):
            return

        self.sample_params = sample_params_fn
        self.pnet = pnet
        self.sampling_distribution_fn = sampling_distribution_fn
        self.canonicalize_fn = canonicalize_fn
        self.decanonicalize_interest_fn = decanonicalize_interest_fn
        self.num_unknown_param = num_unknown_param
        self.num_known_param = num_known_param

    ###########################################################################
    #
    #  Methods overridden from _SimulatorNet
    #
    ###########################################################################

    def simulate_training_data(
            self,
    ) -> Tuple[NetInputBlob, NetTargetBlob]:

        n = self.batch_size
        params = self.sample_params(n)
        estimates = self.sampling_distribution_fn(params)
        target_z = tf.random.normal((n,))

        known_params = self.known_params(params)

        inputs = (estimates, known_params, target_z)
        outputs = (estimates, known_params, target_z)

        return inputs, outputs

    def get_loss(
            self,
            net_outputs: NetOutputBlob,
            target_outputs: NetTargetBlob,
    ) -> ttf.float32:

        estimates, known_params, target_z = target_outputs
        interest = self.interest_from_net_outputs(net_outputs, estimates)
        z = self.pnet.z_from_interest(estimates, interest, known_params)
        squared_errors = tf.square(z - target_z)
        return tf.reduce_mean(squared_errors)

    ###########################################################################
    #
    #  Tensorflow members
    #
    ###########################################################################

    def net_inputs(
            self,
            net_input_blob: NetInputBlob,
    ) -> List[Tensor2[tf32, Samples, NetInputs]]:

        estimates, known_params, target_z = net_input_blob
        estimates_canonical, known_params_canonical, _ = \
            self.canonicalize_fn(estimates, known_params, None)
        ins = tf.concat([
            estimates_canonical,
            known_params_canonical,
            tf.transpose([target_z])
        ], axis=1)

        return [ins]

    def interest_from_net_outputs(
            self,
            net_outputs: NetOutputBlob,
            stats: Tensor2[tf32, Samples, Stats],
    ) -> Tensor1[tf32, Samples]:

        interest_canonical = net_outputs[:, 0]
        interest = self.decanonicalize_interest_fn(interest_canonical, stats)
        return interest

    def known_params(
            self,
            params: Tensor2[tf32, Samples, Params],
    ) -> Tensor2[tf32, Samples, KnownParams]:

        return known_params_from_params(params, self.num_unknown_param,
                                                self.num_known_param)          # type: ignore

    def interest_from_net_input_blob(
            self,
            net_input_blob: NetInputBlob,
    ) -> Tensor1[tf32, Samples]:

        net_outputs = self.call_tf(net_input_blob)
        estimates, known_params, target_z = net_input_blob
        interest = self.interest_from_net_outputs(net_outputs, estimates)
        return interest

    def ci(
            self,
            estimates: Tensor2[tf32, Samples, Stats],
            known_params: Tensor2[tf32, Samples, KnownParams],
            target_p: Tensor1[tf32, Samples],
    ) -> Tuple[Tensor1[tf32, Samples], Tensor1[tf32, Samples]]:

        target_z_pos = self.pnet.positive_z_from_p(target_p)
        lower = self.interest_from_net_input_blob(
            (estimates, known_params, target_z_pos),
        )
        upper = self.interest_from_net_input_blob(
            (estimates, known_params, -target_z_pos),
        )

        return lower, upper
