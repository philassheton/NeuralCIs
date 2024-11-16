import tensorflow as tf
from tensorflow.python.eager.def_function import Function as TFFunction        # type: ignore
import numpy as np

from neuralcis import common
from neuralcis import sampling
from neuralcis._param_sampler import _ParamSampler
from neuralcis._p_net import _PNet
from neuralcis._ci_net import _CINet
from neuralcis._data_saver import _DataSaver
from neuralcis.common import HAT

# for typing
from typing import Tuple, Union, Callable, List, Sequence, Dict, Optional
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2
from tensor_annotations.tensorflow import float32 as tf32
from neuralcis.common import Samples, Estimates, Params
from neuralcis.distributions import Distribution


def no_transform(estimates, params):
    return estimates, params


class NeuralCIs(_DataSaver):
    """Train neural networks that compute *p*-values and confidence intervals.

    The following methods must also be implemented:

    :param sampling_distribution_fn: Generate samples from the sampling
        distribution.  This function will be fed 1D Tensorflow Tensors, where
         the elements of each Tensor at a given index represent the
         parameter values at one given sample, and should return 1D Tensors
         of the same length, in the same order.  The return value should be
         a dict, whose values are estimates of the parameters (except for
         any parameters that are known *a priori*), and whose keys are the
         names of those parameters (and exactly the same as the names used
         in the function signature).  See example below.
    :param transform_on_params_fn: An optional function that maps the estimate
        and param tensors (passed as named arguments) to transformed values
        that are expected to give the same p-value.  These transforms should
        be based ONLY on the PARAM values.  Transformed estimate and
        parameters are returned in a dict.

        If provided, `transform_on_estimates_fn` MUST also be provided.

        For example, for a t-test, we could divide all values (except n) by
        our parameter sigma and end up with the same geometry, just rescaled.

        IMPORTANT: This function is only used during training; the
        `transform_on_estimates_fn` is then used during inference.
    :param transform_on_estimates_fn: An optional function that maps the
        estimate and param tensors (passed as named arguments) to transformed
        values that are expected to give the same p-value.  This function
        should base these transforms ONLY on the ESTIMATE values.  Transformed
        estimate and parameters are returned in a dict.

        If provided, `transform_on_params_fn` MUST also be provided.

        For example, for a t-test, we could divide all values (except n) by
        our estimated of sigma and end up with the same geometry, just
        rescaled.

        IMPORTANT: This function is only used during inference; the
        `transform_on_params_fn` is then used during inference.
    :param foldername: Optional string; will load network weights from a
        previous training session.
    :param train_initial_weights:  A bool (default True) that controls whether
        the layers of the underlying nets are optimized to maintain standard
        deviations of input and output.  This is particularly useful when using
        e.g. monotonic layers, whose values can easily blow up without careful
        initialisation.
    :param **param_distributions: For each parameter to the
        sampling_distribution_fn a parameter sampling distribution object
        needs to be passed in by name (same name as in the sampling function).

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
        return {"mu": mu_hat}

    cis = neuralcis.NeuralCIs(
        normal_sampling_fn,
        mu=   neuralcis.Uniform(  -2., 2.  ),
        sigma=neuralcis.LogUniform(.1, 10. ),
        n=    neuralcis.LogUniform(3., 300.)
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
            contrast_fn: Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Tensor1[tf32, Samples]
            ],
            transform_on_params_fn: Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Dict["str", Tensor1[tf32, Samples]]
            ] = None,
            transform_on_estimates_fn: Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Dict["str", Tensor1[tf32, Samples]],
            ] = None,
            foldername: Optional[str] = None,
            train_initial_weights: bool = True,
            network_setup_args: Optional[Dict] = None,
            **param_distributions: Distribution,
    ) -> None:

        if foldername is not None:
            train_initial_weights = False

        if ((transform_on_params_fn is None) !=
            (transform_on_estimates_fn is None)):
            raise Exception("If you provide a transform_on_params_fn, you MUST"
                            " also provide a transform_on_estimates_fn and"
                            " vice versa.")

        self.sampling_distribution_fn = self.tf_fun(sampling_distribution_fn)
        self.contrast_fn = self.tf_fun(contrast_fn)
        self.transform_on_params_fn = self.tf_fun(transform_on_params_fn)
        self.transform_on_estimates_fn = self.tf_fun(transform_on_estimates_fn)

        (
            self.param_names_in_net_order,
            self.estimate_names,
            self.sim_to_net_order,
            self.net_to_sim_order,
            self.param_dists_in_net_order,
            self.estimate_dists_in_net_order,
        ) = self._align_simulation_params(param_distributions)

        self.num_param = len(self.param_dists_in_net_order)
        self.num_estimate = len(self.estimate_dists_in_net_order)
        self.num_unknown_param = self.num_estimate
        self.num_known_param = self.num_param - self.num_unknown_param

        self.net_to_contrast_order = self._align_contrast_fn_params()

        (
            self.has_transform,
            self.net_to_transform_order,
            self.fn_to_net_estimates_order,
            self.fn_to_net_params_order,
            num_params_remaining_after_transform,
        ) = self._align_transform_by_params_fn_inputs()

        estimates_min_and_max_std_uniform = tf.stack([
            dist.min_and_max_std_uniform
            for dist in self.estimate_dists_in_net_order
        ], axis=0)
        estimates_min_and_max = sampling.uniform_from_std_uniform(
            estimates_min_and_max_std_uniform,
            common.PARAMS_MIN, common.PARAMS_MAX
        )

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
            self.num_unknown_param,
            self.num_known_param,
            self.known_param_indices,
            train_initial_weights=train_initial_weights,
            **network_setup_args,
        )
        self.pnet = _PNet(
            self._sampling_dist_net_interface,
            self._contrast_fn_net_interface,
            self._transform_on_params_fn_net_interface,
            self.num_unknown_param,
            self.num_known_param,
            self.known_param_indices,
            num_params_remaining_after_transform,
            self.param_sampler,
            train_initial_weights=train_initial_weights,
            **network_setup_args,
        )
        self.cinet = _CINet(
            self.pnet,
            self._sampling_dist_net_interface,
            self.param_sampler.sample_params,
            self.num_param,
            self.known_param_indices,
            train_initial_weights=train_initial_weights,
            **network_setup_args,
        )

        _DataSaver.__init__(
            self,
            {"paramsampnet": self.param_sampler,
             "pnet": self.pnet,
             "cinet": self.cinet},
        )

        if foldername is not None:
            _DataSaver.load(self, foldername, common.CIS_FILE_START)

    @staticmethod
    def tf_fun(
            func: Union[None, Callable, TFFunction]
    ) -> Optional[TFFunction]:

        if func is None:
            return None
        elif isinstance(func, TFFunction):
            return func
        else:
            return tf.function(func)

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
        self.cinet.fit(*args, **kwargs)

    def values_grid(
            self,
            value_names: Sequence[str] = ("p",),
            return_also_axes: Sequence[str] = (),
            **estimates_and_params: Union[np.ndarray, tf.Tensor, float],
    ) -> List[np.ndarray]:

        """Calculate the p-value across a grid of estimates and/or params.

        For each estimate and param, either a single fixed value or a
        range/sequence of values must be entered via the two dicts, estimates
        and params.  The p-value is then computed at all combinations of each
        of these values.

        :param **estimates_and_params: Named arguments mapping each estimate
         and param name to either a range/sequence of values, or to a single
         fixed value.
        :param value_names: Sequence of strs (default contains only "p");
         list of values to be returned.  Currently also supports "z0", "z1",
         etc., as well as "{estimate_name}_lower" and "{estimate_name}_upper".
        :param return_also_axes: A sequence of str values: the names of the
         axes that should also be returned.  If this is not empty, the return
         type will be a list with these axes first, and the output values
         grid last.  If it is (), a list with only the requested `value_names`
         is returned.
        :return:
        """

        all_names = self.estimate_names + self.param_names_in_net_order
        all_values = [tf.constant(estimates_and_params[n], dtype=tf.float32)
                      for n in all_names]
        all_grids = tf.meshgrid(*all_values)
        shape = all_grids[0].shape
        all_grids_flattened = [tf.reshape(x, [-1]) for x in all_grids]

        estimates_params_flattened = {n: v
                                      for n, v in zip(all_names,
                                                      all_grids_flattened)}
        values_dict = self.ps_and_cis(
            extra_values_names=value_names,
            **estimates_params_flattened,
        )
        values_seq = [np.squeeze(np.reshape(values_dict[n], shape))
                      for n in value_names]

        if len(return_also_axes) > 0:
            estimates_params_grids = {n: g for n, g in zip(all_names,
                                                           all_grids)}
            return_grids = [np.squeeze(estimates_params_grids[n])
                            for n in return_also_axes]
            return return_grids + values_seq
        else:
            return values_seq

    def ps_and_cis(
            self,
            conf_levels: Optional[np.ndarray] = None,
            extra_values_names: Sequence[str] = (),
            apply_transform: bool = True,
            **estimates_and_params: Union[Tensor1[tf32, Samples], np.ndarray],
    ) -> Dict[str, np.ndarray]:

        """Calculate the p-values and confidence intervals for a series of
        novel cases.

        :param **estimates_and_params: A set of named params, giving values
            for the estimates and null hypothesis params (named as per their
            naming in the simulation function).  Each of these may be a
            Tensor, numpy.ndarray or a sequence of floats.
        :param conf_levels: An optional list of floats (default .95).
            Confidence level for each respective  confidence interval.  If
            None, then no confidence interval is computed and only p-values
            are returned.
        :param extra_values_names: An optional sequence of strs, giving
            extra values to be returned from the p-net.  Currently, supports
            "z0", "z1", ...  up to the number of zs but may be expanded later.
        :param apply_transform: A bool, default True.  If False, the
            transform_on_estimates function will be bypassed.  For testing
            purposes only.
        :return: Dict with float values: p-value, lower and upper CI bounds.
        """

        estimates_and_params_tf = {k: tf.constant(v, tf.float32)
                                   for k, v in estimates_and_params.items()}

        if apply_transform:
            estimates_and_params_tf = self._transform_on_estimates(
                **estimates_and_params_tf
            )

        estimates_tf = [estimates_and_params_tf[n]
                        for n in self.estimate_names]
        params_tf = [estimates_and_params_tf[n] for
                     n in self.param_names_in_net_order]

        estimates_net = self._estimates_human_net_order_to_net(*estimates_tf)
        params_net = self._params_human_net_order_to_net(*params_tf)

        # TODO: This should probably live here and be passed down.
        known_params = self.cinet.known_params(params_net)
        if len(extra_values_names) > 0:
            values = self.pnet.p_workings(estimates_net, params_net)
            values = {"p": values["p"].numpy()} | \
                     {k: values[k].numpy() for k in extra_values_names}
        else:
            p = self.pnet.p(estimates_net, params_net)
            values = {'p': p.numpy()}

        if conf_levels is not None:
            # TODO: Contrast is not currently transformed.  Should change that.
            #       (Could actually do that to give it unif probability too!)
            #       And if so, then it would need to be de-transformed here.
            target_p = tf.constant(1. - conf_levels)
            lower, upper = self.cinet.ci(estimates_net, known_params, target_p)

            values["lower"] = lower.numpy()
            values["upper"] = upper.numpy()

        return values

    def p_and_ci(
            self,
            conf_level: float = common.DEFAULT_CONFIDENCE_LEVEL,
            **estimates_and_params: Tensor1[tf32, Samples],
    ) -> Dict[str, float]:

        """Calculate the p-value and confidence interval for a novel case.

        This is the "user-friendly" interface to the network.  Pass in a
        single estimate, null parameter value, and known parameters, and
        it will return p-value, lower bound, upper bound.

        :param: **estimates_and_params, a set of named parameters, all floats,
            giving values for the estimates and null hypothesis params for
            which a single p-value is to be calculated.  Naming should be the
            same as in the simulation function.
        :param conf_level: A float (default .95).  Confidence level for the
            confidence interval.
        :return: Dict with float values: p-value, lower and upper CI bounds.
        """

        estimates_and_params_numpy = {k: np.array([v], dtype=np.float32)
                                      for k, v in estimates_and_params.items()}
        conf_levels = np.array([conf_level], dtype=np.float32)

        ps_and_cis = self.ps_and_cis(conf_levels, **estimates_and_params_numpy)

        p_and_ci = {k: v[0].tolist() for k, v in ps_and_cis.items()}

        return p_and_ci

    def load(self, *args) -> None:

        """Loading weights from a pre-constructed net is now disabled.

        To load a previously saved NeuralCIs object, you need to pass the
        `foldername` to the constructor, when first constructing the net.
        """

        raise Exception("To load a previously saved NeuralCIs object, you "
                        "need to pass a foldername to the constructor.")

    def save(
            self,
            foldername: str,
            *args
    ) -> None:

        """Save weights and neural architectures stored to disk into self.

        NB this does NOT currently save the sampling or contrast functions,
        and these must still be supplied before reloading the network weights.

        :param foldername: A str, the folder in which the weights and
            configurations are to be stored.
        """

        if len(args):
            raise Exception("NeuralCIs only allows you to provide a foldername"
                            " when saving.")

        super().save(foldername, common.CIS_FILE_START)

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
    ) -> Tensor2[tf32, Samples, Estimates]:

        params_human = self._params_net_to_human_in_net_order(params_net)
        params = self._reorder(params_human, self.net_to_sim_order)
        estimates = self.sampling_distribution_fn(*params).values()
        estimates_net = self._estimates_human_net_order_to_net(*estimates)

        return estimates_net

    @tf.function
    def _contrast_fn_net_interface(
            self,
            params_net: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        params_human = self._params_net_to_human_in_net_order(params_net)
        params = self._reorder(params_human, self.net_to_contrast_order)
        contrasts = self.contrast_fn(*params)

        return contrasts

    @tf.function
    def _transform_on_params_fn_net_interface(
            self,
            estimates_net: Tensor2[tf32, Samples, Estimates],
            params_net: Tensor2[tf32, Samples, Params],
    ) -> Tuple[Tensor2[tf32, Samples, Estimates],
               Tensor2[tf32, Samples, Params]]:

        if not self.has_transform:
            return estimates_net, params_net

        estimates_human = \
                    self._estimates_net_to_human_in_net_order(estimates_net)
        params_human = self._params_net_to_human_in_net_order(params_net)
        inputs = self._reorder(estimates_human + params_human,
                               self.net_to_transform_order)

        outputs = self.transform_on_params_fn(*inputs).values()

        estimates_human = self._reorder(outputs,
                                        self.fn_to_net_estimates_order)
        params_human = self._reorder(outputs, self.fn_to_net_params_order)

        estimates_net = \
                       self._estimates_human_net_order_to_net(*estimates_human)
        params_net = self._params_human_net_order_to_net(*params_human)

        return estimates_net, params_net

    @tf.function
    def _preprocess_params_net_interface(
            self,
            params_net: Tensor2[tf32, Samples, Params],
            known_params_only: bool = False,
    ) -> Tensor2[tf32, Samples, Params]:

        params_human_preprocessed = self._params_net_to_human_in_net_order(
            params_net,
            known_params_only=known_params_only,
            preprocess=True,
        )
        params_net_preprocessed = self._params_human_net_order_to_net(
            *params_human_preprocessed,
            known_params_only=known_params_only,
        )
        return params_net_preprocessed

    def _transform_on_estimates(
            self,
            **estimates_and_params: Tensor1[tf32, Samples],
    ) -> Dict[str, Tensor1[tf32, Samples]]:

        if self.transform_on_estimates_fn is None:
            return estimates_and_params

        untransformed = estimates_and_params
        transformed = self.transform_on_estimates_fn(**untransformed)
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
    #   (1) Before we pass estimates or parameters to the net, we transform
    #       them in such a way that they should be closer to uniform
    #       distributed (e.g. by log-transforming scale variables).
    #
    #   (2) The net assumes a particular order to the parameters, whereas the
    #       inputs to the sampling function could be in any order.  The order
    #       assumed by the net is: (i) parameters to be estimated,
    #       (ii) nuisance parameters and then (iii) known parameters (e.g.
    #       sample size).
    #
    ###########################################################################

    @tf.function
    def _params_net_to_human_in_net_order(
            self,
            params_net: Tensor2[tf32, Samples, Params],
            known_params_only: bool = False,
            preprocess: bool = False,
    ) -> List[Tensor1[tf32, Samples]]:

        if known_params_only:
            num_param = self.num_known_param
            param_dists = self.param_dists_in_net_order[-num_param:]
        else:
            num_param = self.num_param
            param_dists = self.param_dists_in_net_order

        params_net_split = tf.unstack(params_net, num=num_param, axis=1)
        params_human_net_order = [d.from_net(p)
                                  for d, p in zip(param_dists,
                                                  params_net_split)]

        if preprocess:
            params_human_net_order = [d.preprocess(p)
                                      for d, p in zip(param_dists,
                                                      params_human_net_order)]

        return params_human_net_order

    @tf.function
    def _params_human_net_order_to_net(
            self,
            *params_human: Tensor1[tf32, Samples],
            known_params_only: bool = False,
    ) -> Tensor2[tf32, Samples, Params]:

        if known_params_only:
            param_dists = self.param_dists_in_net_order[-self.num_known_param:]
        else:
            param_dists = self.param_dists_in_net_order

        params_net_split = [d.to_net(p)
                            for d, p in zip(param_dists, params_human)]
        params_net = tf.stack(params_net_split, axis=1)

        return params_net

    @tf.function
    def _estimates_net_to_human_in_net_order(
            self,
            estimates_net: Tensor2[tf32, Samples, Estimates],
    ) -> List[Tensor1[tf32, Samples]]:

        estimates_net_split = tf.unstack(estimates_net,
                                         num=self.num_estimate, axis=1)
        estimates_human_net_order = [d.from_net(p) for d, p in
                                     zip(self.estimate_dists_in_net_order,
                                         estimates_net_split)]
        return estimates_human_net_order

    @tf.function
    def _estimates_human_net_order_to_net(
            self,
            *estimates_human: Tensor1[tf32, Samples],
    ) -> Tensor2[tf32, Samples, Estimates]:

        estimates_net_split = [d.to_net(p) for d, p in
                               zip(self.estimate_dists_in_net_order,
                                   estimates_human)]
        estimates_net = tf.stack(estimates_net_split, axis=1)
        return estimates_net

    @tf.function
    def _reorder(self, tensors: List[Tensor1], order: List[int]) \
            -> List[Tensor1]:

        return [tensors[i] for i in order]

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

    def _get_estimates_names(
            self,
            param_distributions_named: Dict[str, Distribution],
    ) -> List[str]:

        test_params = self._generate_params_test_sample(
            param_distributions_named,
            common.BATCH_SIZE,
        )
        estimates = self.sampling_distribution_fn(*test_params)
        estimate_names = list(estimates.keys())
        if not np.all([e.endswith(HAT) for e in estimate_names]):
            raise Exception(f"All estimate names must end with '{HAT}'!!")
        return estimate_names

    def _estimate_names_dehatted(
            self,
            estimate_names: Optional[List[str]] = None,
    ) -> List[str]:

        if estimate_names is None:
            estimate_names = self.estimate_names
        return [e.removesuffix(HAT) for e in estimate_names]

    def _generate_params_test_sample(
            self,
            param_distributions_named: Dict[str, Distribution],
            n: int,
    ) -> List[Tensor1[tf32, Samples]]:

        sim_params = self._get_tf_params(self.sampling_distribution_fn)
        dists = [param_distributions_named[p] for p in sim_params]
        params = [d.from_std_uniform(tf.random.uniform((n,))) for d in dists]
        return params

    @staticmethod
    def _get_tf_params(
            tf_function: TFFunction,
    ) -> List[str]:

        return tf_function.function_spec.arg_names

    def _align_simulation_params(
            self,
            param_distributions_named: dict,
    ) -> Tuple[
        List[str],
        List[str],
        List[int],
        List[int],
        List[Distribution],
        List[Distribution],
    ]:

        sim_order_names = self._get_tf_params(self.sampling_distribution_fn)
        n = len(param_distributions_named)

        estimate_names = self._get_estimates_names(param_distributions_named)
        estimate_names_dehatted = self._estimate_names_dehatted(estimate_names)

        if np.any([p.endswith(HAT) for p in param_distributions_named.keys()]):
            raise Exception(f"None of your param names may end with {HAT}!!")
        if np.any([p.endswith(HAT) for p in sim_order_names]):
            raise Exception(f"None of your simulation params may end with"
                            f" {HAT}!!")
        if not np.all([e.endswith(HAT) for e in estimate_names]):
            raise Exception(f"All of your estimate names MUST end with {HAT}!")

        assert (
            sorted(param_distributions_named.keys()) == sorted(sim_order_names)
        )

        # pull the estimated param(s) to the start to match the convention
        #   within the networks
        unknown_param_indices_in_sim_pars = \
            [sim_order_names.index(e) for e in estimate_names_dehatted]
        known_param_indices_in_sim_pars = \
            [i for i in range(n) if i not in unknown_param_indices_in_sim_pars]
        sim_to_net_order = \
            unknown_param_indices_in_sim_pars + known_param_indices_in_sim_pars

        sorted_inds_and_net_to_sim = sorted(zip(sim_to_net_order, range(n)))
        net_to_sim_order = [x[1] for x in sorted_inds_and_net_to_sim]
        net_order_names = [sim_order_names[i] for i in sim_to_net_order]

        param_transforms_in_net_order = [
            param_distributions_named[i] for i in net_order_names
        ]
        estimate_transforms_in_net_order = [
            param_distributions_named[i] for i in estimate_names_dehatted
        ]

        return (
            net_order_names,
            estimate_names,
            sim_to_net_order,
            net_to_sim_order,
            param_transforms_in_net_order,
            estimate_transforms_in_net_order,
        )

    def _align_contrast_fn_params(
            self,
    ) -> List[int]:

        # note that we only need transforms on the way in: since we look at
        #   each contrast in isolation, and since we only care about how the
        #   derivs are proportioned to each other, any further transform will
        #   only multiply the derivs by a constant term.
        # TODO: Look at whether we might also want to allow distributions for
        #       the contrasts, to keep them in a good range.  (see comment
        #       above).

        fn_order_params = self._get_tf_params(self.contrast_fn)
        net_order_params = self.param_names_in_net_order

        net_to_con_order = [fn_order_params.index(p) for p in net_order_params]

        return net_to_con_order

    def _align_transform_by_params_fn_inputs(
            self,
    ) -> Tuple[bool,
               List[int],
               List[int],
               List[int],
               int]:

        if self.transform_on_params_fn is None:
            return False, [], [], [], self.num_param

        # TODO: there is a lot duplicated here from functions above.  Need to
        #       find a neat framework for this all to work cleanly.  Also this
        #       is very much a quick dirty test-it-out first draft.  Tidy!!
        fn_order_inputs = self._get_tf_params(self.transform_on_params_fn)
        net_order_params = self.param_names_in_net_order
        net_order_estimates = self.estimate_names

        test_inputs = [tf.random.uniform((common.BATCH_SIZE,))
                       for _ in fn_order_inputs]
        test_outputs = self.transform_on_params_fn(*test_inputs)
        fn_order_outputs = [n for n in test_outputs.keys()]

        num_estimate = len(net_order_estimates)
        net_order_params_estimates = (
            {n: i for i, n in enumerate(net_order_estimates)} |
            {n: i + num_estimate for i, n in enumerate(net_order_params)}
        )

        net_to_fn_order = [net_order_params_estimates[n]
                           for n in fn_order_inputs]
        fn_to_net_estimates_order = [fn_order_outputs.index(e)
                                     for e in net_order_estimates]
        fn_to_net_params_order = [fn_order_outputs.index(p)
                                  for p in net_order_params
                                  if p in fn_order_outputs]

        num_params_remaining = len(fn_to_net_params_order)

        return (True,
                net_to_fn_order,
                fn_to_net_estimates_order,
                fn_to_net_params_order,
                num_params_remaining)

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
            (common.SAMPLES_TO_TEST_PARAM_MAPPINGS, self.num_param),
            common.PARAMS_MIN,
            common.PARAMS_MAX,
        )
        params_human = self._params_net_to_human_in_net_order(params_net)
        params_net_again = self._params_human_net_order_to_net(*params_human)
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
        def to_net(value, dist: Distribution):
            if isinstance(value, float):
                value = tf.fill((num_samples,), value)
            else:
                assert isinstance(value, tf.Tensor)
                assert len(value.shape) == 1 and len(value) == num_samples
            return dist.to_net(value)

        known_param_min_values = []
        known_param_max_values = []
        known_param_names_in_net_order = [self.param_names_in_net_order[i]
                                          for i in self.known_param_indices]
        known_param_dists_in_net_order = [self.param_dists_in_net_order[i]
                                          for i in self.known_param_indices]

        for n, d in zip(known_param_names_in_net_order,
                        known_param_dists_in_net_order):

            if n in known_param_ranges:
                min, max = known_param_ranges[n]
                known_param_min_values.append(to_net(min, d))
                known_param_max_values.append(to_net(max, d))
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

        params_human = self._params_net_to_human_in_net_order(params_net)
        params_dict = {n: p for n, p in zip(self.param_names_in_net_order,
                                            params_human)}
        return params_dict
