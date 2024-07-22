import tensorflow as tf
import tensorflow_probability as tfp                                           # type: ignore
from neuralcis._matcher_net import _MatcherNet
from neuralcis._z_net import _ZNet
from neuralcis._coordi_net import _CoordiNet
from neuralcis._estimator_net import _EstimatorNet
from neuralcis._layers import _LinearLayer, _MultiplyerLayer
from neuralcis._layers import _MonotonicLinearLayer
from neuralcis._layers import _MonotonicWithParamsTanhLayer
from neuralcis._data_saver import _DataSaver
from neuralcis import common

# typing
from typing import Callable, Tuple, Dict
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

        neurons_per_hid = common.MONOTONIC_WITH_PARAMS_NEURONS_PER_HIDDEN_LAYER
        first_layer_types = (_MonotonicWithParamsTanhLayer, _LinearLayer)
        hidden_layer_types = (_MonotonicWithParamsTanhLayer, _MultiplyerLayer)
        output_layer_types = (_MonotonicLinearLayer, _LinearLayer)             # TODO: This will also include the params in the final output, but that doesn't stop it being monotonic
        self.znet_constrained = _ZNet(
            self.sampling_distribution_remapped,                               # type: ignore
            self.sample_params,
            contrast_fn,
            self.validation_set_remapped,                                      # type: ignore
            known_param_indices,
            num_hidden_layers=common.MONOTONIC_WITH_PARAMS_HIDDEN_LAYERS,      # TODO: This is a bit of a hack; will not allow modification via kwargs.  Need to change to settings-per-net or net-type.
            num_neurons_per_hidden_layer=neurons_per_hid,                      # TODO: This same as above
            first_layer_type_or_types=first_layer_types,
            hidden_layer_type_or_types=hidden_layer_types,
            output_layer_type_or_types=output_layer_types,
            coords_fn=self.coordinet.call_tf,                                  # type: ignore
            **network_setup_args,
        )

        self.znet_final = _ZNet(
            self.sampling_distribution_fn,                                     # type: ignore
            self.sample_params,
            contrast_fn,
            self.validation_set,
            known_param_indices,
            self.coordinet.call_tf,                                            # type: ignore
            **network_setup_args,
        )
        self.znet_matcher = _MatcherNet(
            self.znet_constrained.call_tf,
            self.simulate_inputs_to_coordi_znet_combo,
            self.znet_final.nets,
            self.znet_final.net_inputs,
        )
        super().__init__(
            filename=filename,
            subobjects_to_save={"znetconstrained": self.znet_constrained,
                                "estimatornet": self.estimatornet,
                                "coordinet": self.coordinet,
                                "znetmatcher": self.znet_matcher,
                                "znetfinal": self.znet_final}
        )

    def fit(self, *args, **kwargs) -> Dict:
        histories = {}
        histories["pnet"] = self.estimatornet.fit(*args, **kwargs)
        histories["coordinet"] = self.coordinet.fit(*args, **kwargs)
        histories["zconstr"] = self.znet_constrained.fit(*args, **kwargs)
        histories["matchernet"] = self.znet_matcher.fit(*args, **kwargs)
        histories["znet"] = self.znet_final.fit(*args, **kwargs)
        return histories

    def compile(self, *args, **kwargs) -> None:
        self.estimatornet.compile(*args, **kwargs)
        self.coordinet.compile(*args, **kwargs)
        self.znet_constrained.compile(*args, **kwargs)
        self.znet_matcher.compile(*args, **kwargs)
        self.znet_final.compile(*args, **kwargs)

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

        zs = self.znet_final.call_tf((estimates, params_null))
        cdf = tfp.distributions.Normal(0., 1.).cdf(zs[:, 0])

        p: Tensor1[tf32, Samples] = 1. - tf.math.abs(cdf*2. - 1.)              # type: ignore

        return p

    @tf.function
    def num_param(self) -> int:
        return self.num_unknown_param + self.num_known_param

    ###########################################################################
    #
    #   Functions for the matcher net
    #
    ###########################################################################

    @tf.function
    def simulate_inputs_to_coordi_znet_combo(
            self,
            n: int,
    ) -> Tuple[
        Tensor2[tf32, Samples, Estimates],
        Tensor2[tf32, Samples, Params],
    ]:

        params = self.sample_params(n)
        estimates = self.sampling_distribution_fn(params)
        return estimates, params
