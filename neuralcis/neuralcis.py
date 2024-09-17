import tensorflow as tf
from tensorflow.python.eager.def_function import Function as TFFunction        # type: ignore
import numpy as np
import os

from neuralcis import common
from neuralcis import sampling
from neuralcis._param_sampling_net import _ParamSamplingNet
from neuralcis._p_net import _PNet
from neuralcis._ci_net import _CINet
from neuralcis._data_saver import _DataSaver

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
    :param transform_fn: An optional function that maps a dict of estimate
        and param tensors to transformed values that are expected to give the
        same p-value.
        For example, for a t-test, we could divide all values by our estimate
        of sigma and end up with the same geometry, just rescaled.  This then
        allows the net to be trained to only accept estimated sigma of 1, and
        all other cases will be rescaled to match this.
        IMPORTANT: This is only used AFTER training, when calling ps_and_cis
        and its derivative functions (such as p_and_ci).
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
            transform_fn: Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Dict["str", Tensor1[tf32, Samples]]
            ] = no_transform,
            foldername: Optional[str] = None,
            train_initial_weights: bool = True,
            network_setup_args: Optional[Dict] = None,
            **param_distributions: Distribution,
    ) -> None:

        if foldername is not None:
            train_initial_weights = False

        if isinstance(sampling_distribution_fn, TFFunction):
            self.sampling_distribution_fn = sampling_distribution_fn
        else:
            self.sampling_distribution_fn = tf.function(
                sampling_distribution_fn,
            )

        if isinstance(contrast_fn, TFFunction):
            self.contrast_fn = contrast_fn
        else:
            self.contrast_fn = tf.function(
                contrast_fn,
            )

        if isinstance(transform_fn, TFFunction):
            self.transform_fn = transform_fn
        else:
            self.transform_fn = tf.function(
                transform_fn
            )

        (
            self.param_names_in_sim_order,
            self.estimate_names,
            self.sim_to_net_order,
            self.net_to_sim_order,
            self.param_dists_in_sim_order,
            self.estimate_dists_in_sim_order,
        ) = self._align_simulation_params(param_distributions)

        (
            self.param_dists_in_contrast_order,
            self.net_to_contrast_order,
        ) = self._align_contrast_fn_params(param_distributions)

        self.num_param = len(self.param_dists_in_sim_order)
        self.num_estimate = len(self.estimate_dists_in_sim_order)

        estimates_min_and_max_std_uniform = tf.stack([
            dist.min_and_max_std_uniform
            for dist in self.estimate_dists_in_sim_order
        ], axis=0)
        estimates_min_and_max = sampling.uniform_from_std_uniform(
            estimates_min_and_max_std_uniform,
            common.PARAMS_MIN, common.PARAMS_MAX
        )

        assert (self._max_error_of_reverse_mapping().numpy() <
                common.ERROR_ALLOWED_FOR_PARAM_MAPPINGS)

        if network_setup_args is None:
            network_setup_args = {}
        num_unknown_param = self.num_estimate
        num_known_param = self.num_param - num_unknown_param
        known_param_indices = [
            i + num_unknown_param for i in range(num_known_param)
        ]
        self.param_sampling_net = _ParamSamplingNet(
            self._sampling_dist_net_interface,
            num_unknown_param, num_known_param, estimates_min_and_max,
            train_initial_weights=train_initial_weights, **network_setup_args
        )
        self.pnet = _PNet(
            self._sampling_dist_net_interface,
            self._contrast_fn_net_interface,
            num_unknown_param,
            num_known_param,
            known_param_indices,
            self.param_sampling_net,
            train_initial_weights=train_initial_weights,
            **network_setup_args,
        )
        self.cinet = _CINet(
            self.pnet,
            self._sampling_dist_net_interface,
            self.param_sampling_net.sample_params,
            self.num_param,
            known_param_indices,
            train_initial_weights=train_initial_weights,
            **network_setup_args,
        )

        _DataSaver.__init__(
            self,
            {"paramsampnet": self.param_sampling_net,
             "pnet": self.pnet,
             "cinet": self.cinet},
        )

        if foldername is not None:
            _DataSaver.load(self, foldername, common.CIS_FILE_START)

    def fit(self, turn_off_gpu: bool = True, *args, **kwargs) -> None:

        """Fit the networks to the simulation.

        This can be run without any parameters, but it is also possible to
        tweak settings by passing in any of the arguments listed below.

        This is a very rough network fitting algorithm in Version 1.0.0.
        Better fitting of the models is a priority in future versions.
        Currently, the default Keras training loop is used with an
        exponentially decreasing learning rate (it will be
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
        :param turn_off_gpu: A bool (default True).  Since we are training
            very small networks, it is generally much much faster to train
            on the CPU.  Setting this true will make any GPUs invisible to
            Tensorflow so that they will not be used.
        """

        self.param_sampling_net.fit(*args, **kwargs)
        self.pnet.fit(*args, **kwargs)
        self.cinet.fit(*args, **kwargs)

    def values_grid(
            self,
            estimates: Dict,
            params: Dict,
            value_names: Sequence[str] = ("p",),
            return_also_axes: bool = True,
    ) -> Sequence[np.ndarray]:

        """Calculate the p-value across a grid of estimates and/or params.

        For each estimate and param, either a single fixed value or a
        range/sequence of values must be entered via the two dicts, estimates
        and params.  The p-value is then computed at all combinations of each
        of these values.

        :param estimates: Dict mapping each estimate name to either a
        range/sequence of values, or to a single fixed value.
        :param params: Dict mapping each param name to either a
        range/sequence of values, or to a single fixed value.
        :param value_names: Sequence of strs (default contains only "p");
        list of values to be returned.  Currently also supports "z0", "z1",
        etc., as well as "{estimate_name}_lower" and "{estimate_name}_upper".
        :param return_also_axes: A bool (default True) that determines whether
        all the axes (estimates, params, etc) are returned.  If False, just
        one grid of p-values are returned.
        :return:
        """

        estimates_values = [tf.constant(x, dtype=tf.float32)
                            for x in estimates.values()]
        params_values = [tf.constant(x, dtype=tf.float32)
                         for x in params.values()]
        all_values = estimates_values + params_values
        all_grids = tf.meshgrid(*all_values)
        shape = all_grids[0].shape
        all_grids_flattened = [tf.reshape(x, [-1]) for x in all_grids]

        num_estimates = len(estimates_values)
        estimates_grids_flattened = {
            k: v for k, v in zip(estimates.keys(),
                                 all_grids_flattened[0:num_estimates])
        }
        params_grids_flattened = {
            k: v for k, v in zip(params.keys(),
                                 all_grids_flattened[num_estimates:])
        }
        values_dict = self.ps_and_cis(
            estimates_grids_flattened,
            params_grids_flattened,
            extra_values_names=value_names,
        )
        values_seq = [np.squeeze(np.reshape(values_dict[n], shape))
                      for n in value_names]

        if return_also_axes:
            return [np.squeeze(gr)
                    for gr, inp in zip(all_grids, all_values)
                    if inp.shape != ()] + values_seq
        else:
            return values_seq

    def ps_and_cis(
            self,
            estimates: Dict[str, np.ndarray],
            params: Dict[str, np.ndarray],
            conf_levels: Optional[np.ndarray] = None,
            extra_values_names: Sequence[str] = (),
    ) -> Dict[str, np.ndarray]:

        """Calculate the p-values and confidence intervals for a series of
        novel cases.

        :param estimates: A dict of lists of floats, all the same length; the
            estimated values of the parameters in each case.
        :param params: A dict of lists of floats, all the same length; for
            each parameter input to the simulation, the values of that
            parameter that corresponds with each estimate (for
            the parameters being estimated, this should be the null values
            under which the *p*-value is to be calculated).  Each should
            be a float.
        :param conf_levels: An optional list of floats (default .95).
            Confidence level for each respective  confidence interval.  If
            None, then no confidence interval is computed and only p-values
            are returned.
        :param extra_values_names: An optional sequence of strs, giving
            extra values to be returned from the p-net.  Currently supports
            "z0", "z1", ...  up to the number of zs but may be expanded later.
        :return: Dict with float values: p-value, lower and upper CI bounds.
        """

        estimates_tf_dict = {k: tf.constant(v, tf.float32)
                             for k, v in estimates.items()}
        params_tf_dict = {k: tf.constant(v, tf.float32)
                          for k, v in params.items()}

        estimates_trans, params_trans = self.transform_fn(
            estimates_tf_dict,
            params_tf_dict
        )

        estimates_tf = [estimates_trans[n] for n in self.estimate_names]
        params_tf = [params_trans[n] for n in self.param_names_in_sim_order]

        estimates_uniform = self._estimates_to_net(*estimates_tf)
        params_uniform = self._params_to_net(*params_tf)

        # TODO: This should probably live here and be passed down.
        known_params = self.cinet.known_params(params_uniform)
        if len(extra_values_names) > 0:
            values = self.pnet.p_workings(estimates_uniform,
                                          params_uniform)
            values = {"p": values["p"].numpy()} | \
                     {k: values[k].numpy() for k in extra_values_names}
        else:
            p = self.pnet.p(estimates_uniform, params_uniform)
            values = {'p': p.numpy()}

        if conf_levels is not None:
            # TODO: Contrast is not currently transformed.  Should change that.
            #       (Could actually do that to give it unif probability too!)
            #       And if so, then it would need to be de-transformed here.
            lower, upper = self.cinet.ci(
                estimates_uniform,
                known_params,
                tf.constant(1. - conf_levels),
            )

            estimate_name = self.estimate_names[0]
            values[estimate_name + "_lower"] = lower.numpy()
            values[estimate_name + "_upper"] = upper.numpy()

        return values

    def p_and_ci(
            self,
            estimates: Dict[str, float],
            params: Dict[str, float],
            conf_level: float = common.DEFAULT_CONFIDENCE_LEVEL,
    ) -> Dict[str, float]:

        """Calculate the p-value and confidence interval for a novel case.

        This is the "user-friendly" interface to the network.  Pass in a
        single estimate, null parameter value, and known parameters, and
        it will return p-value, lower bound, upper bound.

        :param estimates: A dict, the estimated values of the parameters.
        :param params: A dict of floats.  For each parameter input to the
            simulation, the value of that parameter (for
            the parameters being estimated, this should be the null values
            under which the *p*-value is to be calculated).  Each should
            be a float.
        :param conf_level: A float (default .95).  Confidence level for the
            confidence interval.
        :return: Dict with float values: p-value, lower and upper CI bounds.
        """

        estimates_numpy = {k: np.array([v], dtype=np.float32)
                           for k, v in estimates.items()}
        params_numpy = {k: np.array([v], dtype=np.float32)
                        for k, v in params.items()}
        conf_levels = np.array([conf_level], dtype=np.float32)

        ps_and_cis = self.ps_and_cis(estimates_numpy,
                                     params_numpy,
                                     conf_levels)

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
            params_transformed: Tensor2[tf32, Samples, Params],
    ) -> Tensor2[tf32, Samples, Estimates]:

        params = self._params_from_net(params_transformed)
        estimates = self.sampling_distribution_fn(*params).values()
        estimates_transformed = self._estimates_to_net(*estimates)

        return estimates_transformed

    @tf.function
    def _contrast_fn_net_interface(
            self,
            params_transformed: Tensor2[tf32, Samples, Params],
    ) -> Tensor1[tf32, Samples]:

        params = self._contrast_params_from_net(params_transformed)
        contrasts = self.contrast_fn(*params)

        return contrasts

    ###########################################################################
    #
    #  Shuffling data to and from the format the underlying nets use, and the
    #   format used by the user-provided sampling function.  These two formats
    #   differ in two key ways:
    #
    #   (1) The net assumes that the parameters should be sampled uniformly
    #       from [-1, 1].  The net first translates these into standard uniform
    #       values and then uses the user-provided distribution objects to
    #       transform these standard uniform values into the raw parameters.
    #
    #   (2) The net assumes a particular order to the parameters, whereas the
    #       inputs to the sampling function could be in any order.  The order
    #       assumed by the net is: (i) parameters to be estimated,
    #       (ii) nuisance parameters (not yet supported) and then (iii) known
    #       parameters.
    #
    ###########################################################################

    @tf.function
    def _contrast_params_from_net(
            self,
            transformed: Tensor2[tf32, Samples, Params],
    ) -> List[Tensor1[tf32, Samples]]:

        # TODO: it may be possible to factor this into _params_from_net
        #       but just writing a first draft for now to try to get this
        #       to work...
        uniform = self._std_uniform_from_net(transformed)
        uniform_net_order = tf.unstack(uniform, num=self.num_param, axis=1)
        uniform_con_order = self._reorder(uniform_net_order,
                                          self.net_to_contrast_order)

        dist_unif = zip(self.param_dists_in_contrast_order, uniform_con_order)
        params = [dist.from_std_uniform(unif) for dist, unif in dist_unif]

        return params

    @tf.function
    def _params_from_net(
            self,
            transformed: Tensor2[tf32, Samples, Params],
    ) -> List[Tensor1[tf32, Samples]]:

        uniform = self._std_uniform_from_net(transformed)
        uniform_net_order = tf.unstack(uniform, num=self.num_param, axis=1)
        uniform_sim_order = self._reorder(uniform_net_order,
                                          self.net_to_sim_order)
        dist_unif = zip(self.param_dists_in_sim_order, uniform_sim_order)
        params = [dist.from_std_uniform(unif) for dist, unif in dist_unif]

        return params

    @tf.function
    def _params_to_net(
            self,
            *params: Tensor1[tf32, Samples],
    ) -> Tensor2[tf32, Samples, Params]:

        dist_par = zip(self.param_dists_in_sim_order, params)
        unif_sim_order = [dist.to_std_uniform(par) for dist, par in dist_par]
        unif_net_order = self._reorder(unif_sim_order, self.sim_to_net_order)
        params_std_uniform_stacked = tf.stack(unif_net_order, axis=1)
        return self._std_uniform_to_net(params_std_uniform_stacked)

    @tf.function
    def _estimates_to_net(
            self,
            *estimates: Tensor1[tf32, Samples],
    ) -> Tensor2[tf32, Samples, Estimates]:

        # the net is organised in the same order as the estimates so no need
        #   for a reordering (unlike with the params)
        dist_est = zip(self.estimate_dists_in_sim_order, estimates)
        uniform = [dist.to_std_uniform(est) for dist, est in dist_est]
        uniform_stacked = tf.stack(uniform, axis=1)
        return self._std_uniform_to_net(uniform_stacked)

    @tf.function
    def _reorder(self, tensors: List[Tensor1], order: List[int]) \
            -> List[Tensor1]:

        return [tensors[i] for i in order]

    @staticmethod
    def _tensor1_first_elem_to_float(
            tensor: Tensor1[tf32, Samples],
    ) -> float:

        return float(tensor.numpy()[0])

    @tf.function
    def _std_uniform_to_net(
            self,
            std_uniform: Tensor2[tf32, Samples, Union[Estimates, Params]],
    ) -> Tensor2[tf32, Samples, Union[Estimates, Params]]:

        return sampling.uniform_from_std_uniform(
            std_uniform, common.PARAMS_MIN, common.PARAMS_MAX
        )

    @tf.function
    def _std_uniform_from_net(
            self,
            transformed: Tensor2[tf32, Samples, Union[Estimates, Params]],
    ) -> Tensor2[tf32, Samples, Union[Estimates, Params]]:

        return sampling.uniform_to_std_uniform(
            transformed, common.PARAMS_MIN, common.PARAMS_MAX
        )

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
            param_distributions_named, 2
        )
        estimates = self.sampling_distribution_fn(*test_params)
        return list(estimates.keys())

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
        List[Distribution]
    ]:

        sim_order_names = self._get_tf_params(self.sampling_distribution_fn)
        n = len(param_distributions_named)

        estimate_names = self._get_estimates_names(param_distributions_named)

        assert (
            sorted(param_distributions_named.keys()) == sorted(sim_order_names)
        )

        # pull the estimated param(s) to the start to match the convention
        #   within the networks
        good_param_indices_in_sim_pars = \
            [sim_order_names.index(e) for e in estimate_names]
        known_param_indices_in_sim_pars = \
            [i for i in range(n) if i not in good_param_indices_in_sim_pars]
        sim_to_net_order = \
            good_param_indices_in_sim_pars + known_param_indices_in_sim_pars

        sorted_inds_and_net_to_sim = sorted(zip(sim_to_net_order, range(n)))
        net_to_sim_order = [x[1] for x in sorted_inds_and_net_to_sim]

        param_transforms_in_sim_order = [
            param_distributions_named[i] for i in sim_order_names
        ]
        estimate_transforms_in_sim_order = [
            param_distributions_named[i] for i in estimate_names
        ]

        return (
            sim_order_names,
            estimate_names,
            sim_to_net_order,
            net_to_sim_order,
            param_transforms_in_sim_order,
            estimate_transforms_in_sim_order
        )

    def _align_contrast_fn_params(
            self,
            param_distributions_named: dict,
    ) -> Tuple[List[Distribution], List[int]]:

        # note that we only need transforms on the way in: since we look at
        #   each contrast in isolation, and since we only care about how the
        #   derivs are proportioned to each other, any further transform will
        #   only multiply the derivs by a constant term.
        # TODO: Look at whether we might also want to allow distributions for
        #       the contrasts, to keep them in a good range.  (see comment
        #       above).

        fn_order_params = self._get_tf_params(self.contrast_fn)
        sim_order_params = self.param_names_in_sim_order
        net_order_params = [sim_order_params[i] for i in self.sim_to_net_order]

        net_to_con_order = [fn_order_params.index(p) for p in net_order_params]

        param_transforms_in_con_order = [
            param_distributions_named[i] for i in fn_order_params
        ]

        return param_transforms_in_con_order, net_to_con_order

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
        test_params = tf.random.uniform(
            (common.SAMPLES_TO_TEST_PARAM_MAPPINGS, self.num_param),
            common.PARAMS_MIN,
            common.PARAMS_MAX,
        )
        as_params = self._params_from_net(test_params)
        as_uniform = self._params_to_net(*as_params)
        errors = tf.math.abs(as_uniform - test_params)                         # type: ignore

        return tf.math.reduce_max(errors)

    ###########################################################################
    #
    #  Extra helper functions
    #
    ###########################################################################

    def sample_params(
            self,
            n: int,
    ) -> Dict[str, Tensor1[tf32, Samples]]:

        params_tensor_in_net_form = self.param_sampling_net.sample_params(n)
        params_in_sim_form = self._params_from_net(params_tensor_in_net_form)
        param_values = {n: p for n, p in zip(self.param_names_in_sim_order,
                                             params_in_sim_form)}

        return param_values
