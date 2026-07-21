import tensorflow as tf
from tensorflow.python.eager.def_function import Function as TFFunction        # type: ignore

import numpy as np
import collections

from . import common
from ._param_sampler import _ParamSampler
from ._p_net import _PNet
from ._ci_net import _CINet
from ._neuralcis_kwargs import _NeuralCIsKWArgs
from ._data_saver import _DataSaver
from .common import FULL, TESTING, INFERENCE

# for typing
from typing import Tuple, Union, Callable, List, Sequence, Dict, Optional
from typing import TypeVar, Type
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2
from tensor_annotations.tensorflow import float32 as tf32
from .common import Samples, Stats, Params, KnownParams, UnknownParams
from .variables import Variable
import tensor_annotations.tensorflow as ttf


T = TypeVar("T", bound="NeuralCIs")


class NeuralCIs(_DataSaver):
    """Train neural networks that compute *p*-values and confidence intervals.

    The following methods must also be implemented:

    :param sampling_distribution_fn: Generate samples from the sampling
        distribution.  This function will be fed 1D Tensorflow Tensors, where
        the elements of each Tensor at a given index represent the
        parameter values at one given sample, and should return 1D Tensors
        of the same length, each in the same order.  The return value should
        be a dict, whose values are the statistics from a randomly generated
        sample.  The statistics must have different names than the parameters
        (e.g. if there is a parameter called 'mu', you may have a statistic
        called 'mu_hat', but not called 'mu').  See example below.
    :param interest_fn:  This function will be fed the same 1D Tensorflow
        Tensors as the sampling_distribution_fn, and should compute from
        those parameters, the parameter value to be estimated.  See example
        below.
    :param unknown_param_names:  A list or tuple of strs.  Gives the names of
        params that should match those in the arguments of the sampling and
        interest functions, which are NOT known a priori and therefore must
        be either estimated or removed as nuisance parameters.  (For example,
        for a t-test, this might be ['mu', 'sigma'].
    :param stat_names:  A list or tuple of strs.  Gives the names of
        estimates that are returned by the sampling_distribution_fn. (For
        example, for a t-test, this might be ['mu_hat', 'sigma_hat'].)
    :param known_param_names:  An optional list or tuple of strs.  Gives the
        names any of the params (should match those in the arguments of the
        sampling and interest functions) which ARE known a priori and
        therefore can simply be conditioned upon. (For example, for a t-test,
        this might be ['n'].)
    :param transform_on_stats_fn: An optional function that maps the
        stat and param tensors (passed as named arguments) to transformed
        values that are expected to give the same p-value.  This function
        should base these transforms ONLY on the STAT values.  Transformed
        stat and parameters are returned in a dict.  Each may be either a
        Tensor, or a Python float (for any statistics that have been made
        constant during the transform; for example, in a t-test, we might
        divide all values by sigma_hat; in this case, sigma_hat is equal to
        one and we might have {'sigma_hat': 1.0, 'sigma': sigma / sigma_hat}.

        For example, for a t-test, we could divide all values (except n) by
        our estimates of sigma and end up with the same geometry, just
        rescaled.
    :param transform_on_stats_stat_names: An optional list or tuple of strs.
        Required if transform functions are supplied.  Gives the names of
        statistics as returned by the transform function.  (Parameters are
        currently assumed to be returned under the same names, but this may
        be relaxed in later versions.)
    :param param_sampling_regularize_jitter_multiply: An optional float
        (default 1) which can be used to jitter the samples in the param
        sampling Jeffreys prior estimate.  See
        param_sampling_regularize_jitter_add.
    :param param_sampling_regularize_jitter_add: An optional float (default 0)
        which controls the magnitude of random jitter added to samples when
        estimating the Fisher information for the Jeffreys prior.  This is
        the standard deviation of random jitter added.  It is also possible
        to add an amount of random jitter that is a multiple of the SD in each
        respective direction using param_smapling_regularize_jitter_multiply.
    :param foldername: Optional string; will load network weights from a
        previous training session.
    :param train_initial_weights:  A bool (default True) that controls whether
        the layers of the underlying nets are optimized to maintain standard
        deviations of input and output.  This is particularly useful when using
        e.g. monotonic layers, whose values can easily blow up without careful
        initialisation.
    :param **param_distributions: For each parameter to the
        sampling_distribution_fn a parameter Variable definition object
        needs to be passed in by name (same names as used above).

    Once an instance of the new class is instantiated, the following members
    allow the model to be fit, and for p-values and confidence intervals to
    be calculated for a new observation:

    :func fit:  Fit the networks to the simulation.
    :func p_and_ci: Calculate *p*-value and confidence interval for a novel
        observation.

    Example:

    import tensorflow as tf
    import neuralcis

    def normal_sampling_fn(mu, sigma, n):
        std_normal = tf.random.normal(tf.shape(mu))
        mu_hat = std_normal * sigma / tf.math.sqrt(n) + mu
        return {"mu_hat": mu_hat}

    def interest_fn(mu, sigma, n):
        return mu

    cis = neuralcis.NeuralCIs(
        normal_sampling_fn,
        interest_fn,
        unknown_param_names=['mu'],
        known_param_names=['sigma', 'n'],
        stat_names=['mu_hat'],
        mu=neuralcis.Location(-2., 2.),
        sigma=neuralcis.Scale(.1, 10.),
        n=neuralcis.SampleSize(3., 300.),
    )

    cis.fit()
    print(cis.p_and_ci(1.96, mu=0., sigma=4., n=16.))
    """

    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Dict["str", Tensor1[tf32, Samples]]
            ],
            interest_fn: Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Tensor1[tf32, Samples]
            ],
            estimates_fn: Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Dict["str", Tensor1[tf32, Samples]]
            ],
            unknown_param_names: Sequence[str],
            stat_names: Sequence[str],
            known_param_names: Sequence[str] = (),
            transform_on_stats_fn: Optional[Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Dict["str", Tensor1[tf32, Samples]],
            ]] = None,
            transform_on_stats_stat_names: Optional[Sequence[str]] = None,
            param_sampling_regularize_jitter_multiply: float = 0.,
            param_sampling_regularize_jitter_add: float = 0.,
            train_initial_weights: bool = True,
            profile: str = FULL,                                               # If you want a more minimal setup, "testing" is much lighter and "inference" even lighter still
            network_setup_args: Optional[Dict] = None,
            optional_data_to_store: Optional[Dict] = None,
            **variable_defs: Variable,
    ) -> None:

        if transform_on_stats_fn is None:
            if (transform_on_stats_stat_names is not None
                and len(transform_on_stats_stat_names) > 0):
                raise Exception(f"Your transform_on_stats_stat_names must be"
                                f" either empty or None if you do not enter"
                                f" a transform_on_stats_fn!!  You entered"
                                f" {transform_on_stats_stat_names}.")
            transform_on_stats_stat_names = stat_names

        # store input arguments in a format suitable for serialization:
        self.kwargs = _NeuralCIsKWArgs(
            variable_defs,
            sampling_distribution_fn,
            interest_fn,
            estimates_fn,
            unknown_param_names,
            stat_names,
            known_param_names,
            transform_on_stats_fn,
            transform_on_stats_stat_names,
            param_sampling_regularize_jitter_multiply,
            param_sampling_regularize_jitter_add,
            profile,
            network_setup_args,
            optional_data_to_store,
        )

        # TODO: look at adding variable defs for interest also
        # TODO: look at allowing estimates also to have different variable defs
        #       after transform

        self.num_unknown_param = len(self.kwargs.unknown_param_names)
        self.num_known_param = len(self.kwargs.known_param_names)
        self.num_stat = len(self.stat_names())

        self.has_transform = not self.kwargs.transform_on_stats_fn.is_none()

        # TODO: Add checks for other functions too.
        self._check_simulation_names()
        self._check_transform_on_stats_fn_names()
        self._check_names_are_not_shared()

        estimates_min_and_max = tf.stack([
            self.variable_defs()[param].estimates_box_min_and_max_net
            for param in self.param_names(unknown_params=True,
                                          known_params=False)
        ], axis=0)

        assert (self._max_error_of_reverse_mapping().numpy() <
                common.ERROR_ALLOWED_FOR_PARAM_MAPPINGS)

        if network_setup_args is None:
            network_setup_args = {}
        self.known_param_indices = [
            i + self.num_unknown_param for i in range(self.num_known_param)
        ]
        self.param_sampler = _ParamSampler(
            estimates_min_and_max,
            self._sampling_dist_net_interface,
            self._preprocess_params_net_interface,
            self._param_is_valid_net_interface,
            self._estimates_fn_net_interface,
            self.num_stat,
            self.num_unknown_param,
            self.num_known_param,
            self.known_param_indices,
            profile,
            train_initial_weights=train_initial_weights,
            regularize_jitter_multiply=\
                                  param_sampling_regularize_jitter_multiply,
            regularize_jitter_add=param_sampling_regularize_jitter_add,
            **network_setup_args,
        )
        self.pnet = _PNet(
            self._sampling_dist_net_interface,
            self._interest_fn_net_interface,
            self._transform_on_stats_fn_net_interface,
            self.num_stat,
            self.num_unknown_param,
            self.num_known_param,
            self.known_param_indices,
            len(self.kwargs.transform_on_stats_stat_names),
            self.param_sampler,
            profile,
            train_initial_weights=train_initial_weights,
            **network_setup_args,
        )
        _DataSaver.__init__(
            self,
            {"paramsampnet": self.param_sampler,
             "pnet": self.pnet},
        )

    @staticmethod
    def wrap_up_kwargs(**kwargs):
        return kwargs

    #@tf.function
    def param_names(
            self,
            unknown_params: bool = True,
            known_params: bool = True,
    ) -> List[str]:

        param_names = []
        if unknown_params:
            param_names += self.kwargs.unknown_param_names
        if known_params:
            param_names += self.kwargs.known_param_names
        return param_names

    # @tf.function
    def num_param(
            self,
            known_params_only: bool = False
    ) -> int:

        if known_params_only:
            return self.num_known_param
        else:
            return self.num_unknown_param + self.num_known_param

    def stat_names(self) -> List[str]:
        return list(self.kwargs.stat_names)

    def stat_names_transformed(self) -> List[str]:
        return list(self.kwargs.transform_on_stats_stat_names)

    def stat_param_names(self) -> List[str]:
        return self.stat_names() + self.param_names()

    def variable_defs(self) -> Dict[str, Variable]:
        return self.kwargs.variable_defs

    def defined_vars(self) -> List[str]:
        return list(self.kwargs.variable_defs.keys())

    def fit(self, *args, **kwargs) -> None:

        """Fit the networks to the simulation.

        This can be run without any parameters, but it is also possible to
        tweak settings by passing in any of the arguments listed below.

        This is a very rough network fitting algorithm in Version 1.0.0.
        Better fitting of the models is a priority in future versions.
        Currently, the default Keras training loop is used with an
        exponentially decreasing learning rate; it will be
        decreased every epoch such that it halves every
        `learning_rate_half_life_epochs` epochs.

        Training is run by default using the Adam algorithm with Nesterov
        gradients; this can be tweaked using the compile method.  Nesterov
        gradients are rather useful in this first cut version of the net,
        where invertabilty is forced by punitive gradients (so the Nesterov
        gradients help the optimisation to avoid jumping into the "danger
        zone").  But, in future cuts where invertability might be built-in,
        they might not be necessary.

        :param steps_per_epoch:  An int (default 1000).  Number of steps
            before learning rate is decreased.
        :param epochs:  An int (default 50).  Number of epochs to run.
        :param verbose: An int or string (default 'auto').  See docs for
            tf.keras.Model.fit.
        :param learning_rate_initial: A float (default .05).  Learning rate
            for the first epoch.
        :param learning_rate_half_life_epochs: An int (default 4).  Learning
            rate will halve each time this number of epochs has passed.
        :param callbacks: An array of callbacks to be used during training.
        """

        self.param_sampler.fit(*args, **kwargs)
        self.pnet.fit(*args, **kwargs)

    def values_grid(
            self,
            value_names: Sequence[str] = ("p",),
            return_also_axes: Sequence[str] = (),
            **stats_and_params: Union[np.ndarray, tf.Tensor, float],
    ) -> List[np.ndarray]:

        """Calculate the p-value across a grid of stats and/or params.

        For each stat and param, either a single fixed value or a
        range/sequence of values must be entered as a named argument. The
        p-value is then computed at all combinations of each of these values.

        :param **stats_and_params: Named arguments mapping each stat
         and param name to either a range/sequence of values, or to a single
         fixed value.
        :param value_names: Sequence of strs (default contains only "p");
         list of values to be returned.  Currently also supports "z0", "z1",
         etc., as well as "{stat_name}_lower" and "{stat_name}_upper".
        :param return_also_axes: A sequence of str values: the names of the
         axes that should also be returned.  If this is not empty, the return
         type will be a list with these axes first, and the output values
         grid last.  If it is (), a list with only the requested `value_names`
         is returned.
        :return:
        """

        all_names = self.stat_names() + self.param_names()
        all_values = [tf.constant(stats_and_params[n], dtype=tf.float32)
                      for n in all_names]
        all_grids = tf.meshgrid(*all_values)
        shape = all_grids[0].shape
        all_grids_flattened = [tf.reshape(x, [-1]) for x in all_grids]

        stats_params_flattened = {n: v
                                  for n, v in zip(all_names,
                                                  all_grids_flattened)}
        values_dict = self.ps_and_cis(
            extra_values_names=value_names,
            **stats_params_flattened,
        )
        values_seq = [np.squeeze(np.reshape(values_dict[n], shape))
                      for n in value_names]

        if len(return_also_axes) > 0:
            stats_params_grids = {n: g for n, g in zip(all_names,
                                                       all_grids)}
            return_grids = [np.squeeze(stats_params_grids[n])
                            for n in return_also_axes]
            return return_grids + values_seq
        else:
            return values_seq

    def ps_and_cis(
            self,
            conf_levels: Optional[np.ndarray] = None,
            extra_values_names: Sequence[str] = (),
            **stats_and_params: Union[Tensor1[tf32, Samples], np.ndarray],
    ) -> Dict[str, np.ndarray]:

        """Calculate the p-values and confidence intervals for a series of
        novel cases.  (CONFIDENCE INTERVALS HAVE BEEN TEMPORARILY REMOVED.)

        :param **stats_and_params: A set of named params, giving values
            for the stats and null hypothesis params (named as per their
            naming in the simulation function).  Each of these may be a
            Tensor, numpy.ndarray or a sequence of floats.
        :param conf_levels: An optional list of floats (default .95).
            Confidence level for each respective  confidence interval.  If
            None, then no confidence interval is computed and only p-values
            are returned.
        :param extra_values_names: An optional sequence of strs, giving
            extra values to be returned from the p-net.  Currently, supports
            "z0", "z1", ...  up to the number of zs but may be expanded later.
        :return: Dict with float values: p-value, lower and upper CI bounds.
        """

        stats_and_params_tf = {n: tf.constant(v, tf.float32)
                               for n, v in stats_and_params.items()}

        stats_human = {n: stats_and_params_tf[n] for n in self.stat_names()}
        params_human = {n: stats_and_params_tf[n] for n in self.param_names()}

        stats_net = self._stats_human_to_net(**stats_human)
        params_net = self._params_human_to_net(**params_human)

        if len(extra_values_names) > 0:
            values = self.pnet.p_workings(stats_net, params_net)
            values = {"p": values["p"].numpy()} | \
                     {k: values[k].numpy() for k in extra_values_names}
        else:
            p = self.pnet.p(stats_net, params_net)
            values = {'p': p.numpy()}

        if conf_levels is not None:
            raise Exception("Generation of CIs is temporarily disabled in"
                            " NeuralCIs.  This will hopefully be returned"
                            " to a version very soon.")

        return values

    def p_and_ci(
            self,
            conf_level: float = common.DEFAULT_CONFIDENCE_LEVEL,
            **stats_and_params: Tensor1[tf32, Samples],
    ) -> Dict[str, float]:

        """Calculate the p-value and confidence interval for a novel case.

        This is the "user-friendly" interface to the network.  Pass in a
        single stat, null parameter value, and known parameters, and
        it will return p-value, lower bound, upper bound.

        :param: **stats_and_params, a set of named parameters, all floats,
            giving values for the stats and null hypothesis params for
            which a single p-value is to be calculated.  Naming should be the
            same as in the simulation function.
        :param conf_level: A float (default .95).  Confidence level for the
            confidence interval.
        :return: Dict with float values: p-value, lower and upper CI bounds.
        """

        stats_and_params_numpy = {k: np.array([v], dtype=np.float32)
                                  for k, v in stats_and_params.items()}
        conf_levels = np.array([conf_level], dtype=np.float32)

        ps_and_cis = self.ps_and_cis(conf_levels, **stats_and_params_numpy)

        p_and_ci = {k: v[0].tolist() for k, v in ps_and_cis.items()}

        return p_and_ci

    @classmethod
    def load(
            cls: Type[T],
            foldername,
            profile: Optional[str] = None,                                     # Will default to TESTING if possible, else INFERENCE.
            network_setup_args: Optional[dict] = None,
            network_setup_arg_overrides: Optional[dict] = None,
            **new_kwargs_for_backward_compatibility,
    ) -> T:

        # TODO: Don't load saved data if in inference profile
        kwargs_object = _NeuralCIsKWArgs.load(
            foldername,
            new_kwargs_for_backward_compatibility,
        )
        kwargs = kwargs_object.kwargs()

        saved_profile = kwargs["profile"]
        if profile is None:
            if cls.profile_can_be_extracted_from(TESTING, saved_profile):
                profile = TESTING
            else:
                profile = INFERENCE
        if not cls.profile_can_be_extracted_from(profile, saved_profile):
            raise Exception(f"The net you tried to load was saved with profile"
                            f" {saved_profile} but you are trying to load it"
                            f" using profile {profile}.  There"
                            f" is not enough data saved to be able to load"
                            f" this profile.  Choose a smaller profile.")
        kwargs["profile"] = profile

        if network_setup_args is not None:
            kwargs["network_setup_args"] = network_setup_args
        if network_setup_arg_overrides is not None:
            kwargs["network_setup_args"] = (kwargs["network_setup_args"]
                                            | network_setup_arg_overrides)
        cis = cls(
            train_initial_weights=False,
            **kwargs,
        )
        cis._load_data(foldername, common.CIS_FILE_START, profile)
        return cis

    def save(
            self,
            foldername: str,
            profile: Optional[str] = None,                                     # Defaults to the most detailed possible profile
    ) -> None:

        kwargs = self.kwargs
        net_profile = kwargs.profile
        if profile is None:
            profile = net_profile
        if profile != net_profile:
            if not self.profile_can_be_extracted_from(profile, net_profile):
                raise Exception(f"You are trying to save with profile"
                                f" {profile}, but your net has only has data"
                                f" up to profile {net_profile}.  Please"
                                f" choose a lower profile!!")

        kwargs.save(foldername, profile)
        self._save_data(foldername, common.CIS_FILE_START, profile)

    ###########################################################################
    #
    #  Private members
    #
    #   -- only single underscores because Tensorflow does not support double
    #       underscore private functions.
    #
    ###########################################################################

    @tf.function
    def _sampling_dist_net_interface(
            self,
            params_net: Tensor2[tf32, Samples, Params],
    ) -> Tensor2[tf32, Samples, Stats]:

        params_human = self._params_net_to_human(params_net)
        stats_human = self.kwargs.sampling_distribution_fn(**params_human)
        stats_net = self._stats_human_to_net(**stats_human)

        return stats_net

    @tf.function
    def _interest_fn_net_interface(
            self,
            params_net: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        params_human = self._params_net_to_human(params_net)
        interest_param = self.kwargs.interest_fn(**params_human)

        return interest_param

    @tf.function
    def _estimates_fn_net_interface(
            self,
            stats_net: Tensor2[tf32, Samples, Stats],
            known_params_net: Tensor2[tf32, Samples, KnownParams],
    ) -> Tensor2[tf32, Samples, UnknownParams]:

        stats_human = self._stats_net_to_human(stats_net)
        known_params_human = self._params_net_to_human(known_params_net,
                                                       known_params_only=True)
        inputs_human = stats_human | known_params_human
        estimates_human = self.kwargs.estimates_fn(**inputs_human)
        estimates_net = self._params_human_to_net(**estimates_human,
                                                       unknown_params=True,
                                                       known_params=False)
        return estimates_net

    @tf.function
    def _transform_on_stats_fn_net_interface(
            self,
            stats_net: Tensor2[tf32, Samples, Stats],
            params_net: Tensor2[tf32, Samples, Params],
    ) -> Tuple[Tensor2[tf32, Samples, Stats],
               Tensor2[tf32, Samples, Params]]:

        if not self.has_transform:
            return stats_net, params_net

        stats_human = self._stats_net_to_human(stats_net)
        params_human = self._params_net_to_human(params_net)

        inputs = stats_human | params_human
        outputs = self.kwargs.transform_on_stats_fn(**inputs)
        stats_net = self._stats_transformed_human_to_net(**outputs)
        params_net = self._params_human_to_net(**outputs)

        return stats_net, params_net

    @tf.function
    def _param_is_valid_net_interface(
            self,
            params_net: Tensor2[tf32, Samples, Params],
            known_params_only: bool = False,
    ) -> Tensor2[ttf.bool, Samples, Params]:

        param_names = self.param_names(unknown_params=not known_params_only)
        params_net_split = self._unstack_params_net(params_net,
                                                    known_params_only)
        vars = self.variable_defs()
        is_valid_list = [vars[name].is_within_hard_limits(param)
                         for name, param in zip(param_names, params_net_split)]
        is_valid_tensor = tf.stack(is_valid_list, axis=1)
        return is_valid_tensor

    @tf.function
    def _unstack_params_net(
            self,
            params_net: Tensor2[tf32, Samples, Params],
            known_params_only: bool = False,
    ) -> List[Tensor1[tf32, Samples]]:

        num_param = self.num_param(known_params_only)
        return tf.unstack(params_net, num=num_param, axis=1)

    @tf.function
    def _preprocess_params_net_interface(
            self,
            params_net: Tensor2[tf32, Samples, Params],
            known_params_only: bool = False,
    ) -> Tensor2[tf32, Samples, Params]:

        param_names = self.param_names(unknown_params=not known_params_only)
        params_net_split = self._unstack_params_net(params_net,
                                                    known_params_only)

        vars = self.variable_defs()
        params_net_preprocessed_split = [vars[name].preprocess(param)
                                         for name, param
                                         in zip(param_names, params_net_split)]

        return tf.stack(params_net_preprocessed_split, axis=1)

    def _transform_on_stats(
            self,
            **stats_and_params: Tensor1[tf32, Samples],
    ) -> Dict[str, Tensor1[tf32, Samples]]:

        if self.kwargs.transform_on_stats_fn.is_none():
            return stats_and_params

        untransformed = stats_and_params
        transformed = self.kwargs.transform_on_stats_fn(**untransformed)
        for name, trans in transformed.items():
            if (isinstance(trans, float) or
                    isinstance(trans, tf.Tensor) and len(trans.shape) == 0):
                transformed[name] = tf.fill(untransformed[name].shape, trans)

        return transformed

    ###########################################################################
    #
    #  Shuffling data to and from the format the underlying nets use, and the
    #   format used by the user-provided sampling function.  These two formats
    #   differ in two key ways:
    #
    #   (1) Before we pass stats or parameters to the net, we transform
    #       them in such a way that they should be closer to uniform
    #       distributed (e.g. by log-transforming scale variables).
    #
    #   (2) The net assumes a particular order to the parameters, whereas the
    #       inputs to the sampling function could be in any order.  The order
    #       assumed by the net is: (i) parameters to be statd,
    #       (ii) nuisance parameters and then (iii) known parameters (e.g.
    #       sample size).
    #
    ###########################################################################

    @tf.function
    def _human_to_net(
            self,
            names_in_net_order: Sequence[str],
            **values_human: Tensor1[tf32, Samples],
    ):

        # VERY important that this loops over names_in_net_order and not
        #   over the dict **human, because it must be in the right order!
        vars = self.variable_defs()
        values_net_split = [vars[name].to_net(values_human[name])
                            for name in names_in_net_order]
        values_net = tf.stack(values_net_split, axis=1)
        return values_net

    @tf.function
    def _net_to_human(
            self,
            names_in_net_order: Sequence[str],
            num_param: int,
            values_net: Tensor2,
    ) -> Dict[str, Tensor1[tf32, Samples]]:

        values_net_split = tf.unstack(values_net, num=num_param, axis=1)
        vars = self.variable_defs()
        values_human = {name: vars[name].from_net(values)
                        for name, values in zip(names_in_net_order,
                                                values_net_split)}
        return values_human

    @tf.function
    def _params_net_to_human(
            self,
            params_net: Tensor2[tf32, Samples, Params],
            known_params_only: bool = False,
    ) -> Dict[str, Tensor1[tf32, Samples]]:

        param_names = self.param_names(unknown_params=not known_params_only)
        num_param = self.num_param(known_params_only)
        params_human = self._net_to_human(param_names, num_param, params_net)

        return params_human

    @tf.function
    def _params_human_to_net(
            self,
            unknown_params: bool = True,
            known_params: bool = True,
            **params_human: Tensor1[tf32, Samples],
    ) -> Tensor2[tf32, Samples, Params]:

        param_names = self.param_names(unknown_params, known_params)
        return self._human_to_net(param_names, **params_human)

    @tf.function
    def _stats_net_to_human(
            self,
            stats_net: Tensor2[tf32, Samples, Stats],
    ) -> Dict[str, Tensor1[tf32, Samples]]:

        return self._net_to_human(self.stat_names(),
                                  self.num_stat,
                                  stats_net)

    @tf.function
    def _stats_human_to_net(
            self,
            **stats_human: Dict[str, Tensor1[tf32, Samples]],
    ) -> Tensor2[tf32, Samples, Stats]:

        return self._human_to_net(self.stat_names(), **stats_human)

    @tf.function
    def _stats_transformed_human_to_net(
            self,
            **stats_transformed_human: Tensor1[tf32, Samples],
    ) -> Tensor2[tf32, Samples, Params]:

        return self._human_to_net(self.kwargs.transform_on_stats_stat_names,
                                  **stats_transformed_human)

    @staticmethod
    def _tensor1_first_elem_to_float(
            tensor: Tensor1[tf32, Samples],
    ) -> float:

        return float(tensor.numpy()[0])

    ###########################################################################
    #
    #  Analyse the sampling function and accompanying parameter distribution
    #   functions, to find the order in which parameters are expected by the
    #   sampling distribution function, and which parameters are estimated.
    #
    ###########################################################################

    def _get_stats_names(
            self,
    ) -> List[str]:

        test_params = self._generate_params_test_sample()
        stats = self.kwargs.sampling_distribution_fn(**test_params)
        stat_names = list(stats.keys())
        return stat_names

    def _generate_params_test_sample(
            self,
    ) -> Dict[str, Tensor1[tf32, Samples]]:

        n = common.BATCH_SIZE
        vars = self.kwargs.variable_defs
        net_width = common.PARAMS_MAX - common.PARAMS_MIN
        net = tf.random.uniform((n,)) * net_width + common.PARAMS_MIN
        params = {name: vars[name].from_net(net)
                  for name in self.param_names()}
        return params

    def _check_simulation_names(
            self,
    ) -> None:

        param_names = self.kwargs.sampling_distribution_fn.arg_names()
        stat_names = self._get_stats_names()

        missing = np.setdiff1d(param_names, self.param_names())
        if len(missing):
            raise Exception(f"The following input to your simulation fn cannot"
                            f" be found in either unknown or known params"
                            f" list: {missing}")
        missing = np.setdiff1d(self.kwargs.unknown_param_names, param_names)
        if len(missing):
            raise Exception(f"The following is in your unknown param names,"
                            f" but does not appear as an input to your"
                            f" simulation fn!!  {missing}")
        missing = np.setdiff1d(self.kwargs.known_param_names, param_names)
        if len(missing):
            raise Exception(f"The following is in your known param names,"
                            f" but does not appear as an input to your"
                            f" simulation fn!!  {missing}")
        missing = np.setdiff1d(self.param_names(), self.defined_vars())
        if len(missing):
            raise Exception(f"The following input to your simulation fn cannot"
                            f" be found in the variable definitions!"
                            f" {missing}")
        missing = np.setdiff1d(stat_names, self.stat_names())
        if len(missing):
            raise Exception(f"The following output from your simulation fn"
                            f" cannot be found in stat_names list: {missing}")
        missing = np.setdiff1d(self.stat_names(), stat_names)
        if len(missing):
            raise Exception(f"The following is in your stat names,"
                            f" but does not appear as an output from your"
                            f" simulation fn!!  {missing}")

    def _check_transform_on_stats_fn_names(
            self,
    ) -> None:

        if self.kwargs.transform_on_stats_fn.is_none():
            return

        # Check **inputs** to the function are every single param and stat name
        fn_args = self.kwargs.transform_on_stats_fn.arg_names()

        unexpected = np.setdiff1d(fn_args, self.param_names()
                                           + self.stat_names())
        if len(unexpected):
            raise Exception(f"Your transform_on_stats_fn should only have"
                            f" argument names matching inputs or outputs"
                            f" of the sampling_distribution_fn.  Unexpected:"
                            f" {unexpected}.")
        missing = np.setdiff1d(self.param_names(), fn_args)
        if len(missing):
            raise Exception(f"Your transform_on_stats_fn must take every"
                            f" single param as argument, even if it does not"
                            f" modify it.  Yours is missing: {missing}")
        missing = np.setdiff1d(self.stat_names(), fn_args)
        if len(missing):
            raise Exception(f"Your transform_on_stats_fn must take every"
                            f" single stat as argument, even if it does"
                            f" not modify it.  Yours is missing: {missing}")

        # Now analyse **outputs** of the transform on stats function
        test_inputs = {name: tf.random.uniform((common.BATCH_SIZE,))
                       for name in fn_args}
        test_outputs = self.kwargs.transform_on_stats_fn(**test_inputs)
        output_names = list(test_outputs.keys())

        missing = np.setdiff1d(self.param_names(), output_names)
        if len(missing):
            raise Exception(f"Your transform_on_stats_fn must return every"
                            f" param after the transform.  Missing:"
                            f" {missing}.")
        missing = np.setdiff1d(self.kwargs.transform_on_stats_stat_names,
                               output_names)
        if len(missing):
            raise Exception(f"Your transform_on_stats_fn must return every"
                            f" stat in transform_on_stats_stat_names. "
                            f" Missing: {missing}.")

        expected_outputs = (self.param_names()
                            + self.kwargs.transform_on_stats_stat_names)
        unexpected = np.setdiff1d(output_names, expected_outputs)
        if len(unexpected):
            raise Exception(f"Your transform_on_stats_fn must return only"
                            f" variables with variable definitions. "
                            f" Unexpected: {unexpected}.")
        unexpected = np.setdiff1d(output_names, self.defined_vars())
        if len(unexpected):
            raise Exception(f"Your transform_on_stat_fn must return only"
                            f" variables with variable definitions. "
                            f" Unexpected: {unexpected}.")

    def _check_names_are_not_shared(self):
        all_names = self.param_names() + self.stat_names()
        name_counter = collections.Counter(all_names)
        non_unique = [name for name, count in name_counter.items()
                      if count > 1]
        if len(non_unique):
            raise Exception(f"The following variables were used multiple times"
                            f" (e.g. as both a known and unknown param, or as"
                            f" both a stat and a param, etc): {non_unique}."
                            f"  You may only use each variable in one"
                            f" position!")

    ###########################################################################
    #
    #  Quick check to see how accurately the reverse mapping of the
    #       distribution objects can reconstruct the values entered.  This
    #       also serves as a regtest of sorts (but needs to be part of the
    #       object to stop the user from passing in their own distribution
    #       objects that do not work properly).
    #
    ###########################################################################

    def _max_error_of_reverse_mapping(self) -> Tensor0[tf32]:
        params_net = tf.random.uniform(
            (common.SAMPLES_TO_TEST_PARAM_MAPPINGS, self.num_param()),
            common.PARAMS_MIN,
            common.PARAMS_MAX,
        )
        params_human = self._params_net_to_human(params_net)
        params_net_again = self._params_human_to_net(**params_human)
        errors = tf.math.abs(params_net_again - params_net)

        return tf.math.reduce_max(errors)

    ###########################################################################
    #
    #  Extra helper functions
    #
    ###########################################################################

    def sample_params(
            self,
            num_samples: int,
            outer: bool = False,
            **known_param_ranges: Union[float, Tensor1[tf32, Samples]],
    ) -> Dict[str, Tensor1[tf32, Samples]]:

        # TODO: Tidy this sample_params function.  Too long, needs factoring.
        def resize(value):
            if isinstance(value, float):
                value = tf.fill((num_samples,), value)
            else:
                assert isinstance(value, tf.Tensor)
                assert len(value.shape) == 1 and len(value) == num_samples
            return value

        known_param_min_values = []
        known_param_max_values = []
        vars = self.variable_defs()
        for name in self.kwargs.known_param_names:
            if name in known_param_ranges:
                min, max = known_param_ranges[name]
                known_param_min_values.append(vars[name].to_net(resize(min)))
                known_param_max_values.append(vars[name].to_net(resize(max)))
            else:
                known_param_min_values.append(tf.fill((num_samples,),
                                                      common.PARAMS_MIN))
                known_param_max_values.append(tf.fill((num_samples,),
                                                      common.PARAMS_MAX))

        known_param_min_values = tf.stack(known_param_min_values, axis=1)
        known_param_max_values = tf.stack(known_param_max_values, axis=1)

        if outer:
            params_net = self.param_sampler.sample_params(
                n_inner=0,
                n_outer=num_samples,
                known_mins_outer=known_param_min_values,
                known_maxs_outer=known_param_max_values,
            )
        else:
            params_net = self.param_sampler.sample_params(
                n_inner=num_samples,
                n_outer=0,
                known_mins_inner=known_param_min_values,
                known_maxs_inner=known_param_max_values,
            )

        params_human = self._params_net_to_human(params_net)
        return params_human

    def store_data(
            self,
            main_key: str,
            sub_key: str,
            data_dict: Dict,
    ) -> None:

        self.kwargs.store_data(main_key, sub_key, data_dict)
