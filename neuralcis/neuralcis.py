import tensorflow as tf
from tensorflow.python.eager.def_function import Function as TFFunction        # type: ignore

from neuralcis import common
from neuralcis import sampling
from neuralcis._p_net import _PNet
from neuralcis._ci_net import _CINet
from neuralcis._data_saver import _DataSaver

# for typing
from typing import Tuple, Union, Callable, List, Dict
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2
from tensor_annotations.tensorflow import float32 as tf32
from neuralcis.common import Samples, Estimates, Params
from neuralcis.distributions import Distribution


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
    :param filename: Optional string; will load network weights from a
        previous training session.
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
            filename: str = "",
            **param_distributions: Distribution
    ) -> None:

        if isinstance(sampling_distribution_fn, TFFunction):
            self.sampling_distribution_fn = sampling_distribution_fn
        else:
            self.sampling_distribution_fn = tf.function(
                sampling_distribution_fn
            )

        if isinstance(contrast_fn, TFFunction):
            self.contrast_fn = contrast_fn
        else:
            self.contrast_fn = tf.function(
                contrast_fn
            )

        (
            self.param_names_in_sim_order,
            self.estimate_names,
            self.sim_to_net_order,
            self.net_to_sim_order,
            self.param_dists_in_sim_order,
            self.estimate_dists_in_sim_order
        ) = self._align_simulation_params(param_distributions)

        (
            self.param_dists_in_contrast_order,
            self.net_to_contrast_order
        ) = self._align_contrast_fn_params(param_distributions)

        self.num_param = len(self.param_dists_in_sim_order)
        self.num_estimate = len(self.estimate_dists_in_sim_order)

        assert(self._max_error_of_reverse_mapping().numpy() <
               common.ERROR_ALLOWED_FOR_PARAM_MAPPINGS)

        self.pnet = _PNet(
            self._sampling_dist_net_interface,
            self._contrast_fn_net_interface,
            num_unknown_param=self.num_estimate,
            num_known_param=self.num_param - self.num_estimate
        )
        self.cinet = _CINet(self.pnet,
                            self._sampling_dist_net_interface,
                            self.num_param)

        _DataSaver.__init__(
            self,
            filename,
            {"pnet": self.pnet,
             "cinet": self.cinet})

    def fit(
            self,
            max_epochs: int = common.MAX_EPOCHS,
            minibatch_size: int = common.MINIBATCH_SIZE,
            learning_rate_initial: float = common.LEARNING_RATE_INITIAL,
            divide_after_flattening_for: int =
                    common.DIVIDE_AFTER_FLATTENING_FOR,
            target_validation_loss_sd_p_net: float =
                    common.TARGET_VALIDATION_LOSS_SD_P_NET,
            target_validation_loss_sd_ci_net: float =
                    common.TARGET_VALIDATION_LOSS_SD_CI_NET,
            precompute_optimum_loss: bool = False
    ) -> None:

        """Fit the networks to the simulation.

        This can be run without any parameters, but it is also possible to
        tweak settings by passing in any of the arguments listed below.

        This is a very rough network fitting algorithm in Version 1.0.0.
        Better fitting of the models is a priority in future versions.  The
        training works as follows:

        A number of progressively smaller learning rates are used until the
        error hardly changes from one batch to the next.  Each epoch consists
        of running a fixed number (default 20) of batches of minibatches.
        After each batch, the error is calculated on the validation set.
        After each epoch, the set of errors from the batches are examined: if
        the best loss has failed to be improved on for ten batches, the
        learning rate is halved.  (The argument `divide_after_flattening_for`
        modifies the number of batches it may fail to improve.)   If the
        set of losses from the batches in the epoch have a standard deviation
        less than the target (`target_validation_loss_sd_p_net` or
        `target_validation_loss_sd_ci_net`), training is terminated.

        Training is run using the Adam algorithm with Nesterov gradients.

        :param precompute_optimum_loss:  A bool (default `False`).  If
            True, an approximate value will be computed for the optimum error
            that can be achieved on the validation set.  This makes monitoring
            the progress of training much easier and will hopefully lead to
            some helpful improvements to the training algorithm in future
            versions.
        :param max_epochs:  An int (default 1000).  Maximum number of epochs
            before training quits.
        :param minibatch_size: An int (default 32).  Number of samples per
            minibatch.
        :param learning_rate_initial: A float (default .001).  Learning rate
            for the first epoch(s).
        :param divide_after_flattening_for: An int (default 10).  If training
            has not improved for this many batches, halve the training rate.
        :param target_validation_loss_sd_p_net: A float (default 1).  If the
            validation errors for the batches within the last epoch have a
            standard deviation below this number, terminate training.  This
            applies only to training of the p-net.
        :param target_validation_loss_sd_ci_net: A float (default 1e-6).  Same
            as `target_validation_loss_sd_p_net`, but applied to the CI net.
        """

        self.pnet.fit(
            max_epochs=max_epochs,
            minibatch_size=minibatch_size,
            learning_rate_initial=learning_rate_initial,
            divide_after_flattening_for=divide_after_flattening_for,
            target_validation_loss_sd=target_validation_loss_sd_p_net,
            precompute_optimum_loss=precompute_optimum_loss
        )
        self.cinet.fit(
            max_epochs=max_epochs,
            minibatch_size=minibatch_size,
            learning_rate_initial=learning_rate_initial,
            divide_after_flattening_for=divide_after_flattening_for,
            target_validation_loss_sd=target_validation_loss_sd_ci_net,
            precompute_optimum_loss=precompute_optimum_loss
        )

    def p_and_ci(
            self,
            estimates: Dict[str, float],
            params: Dict[str, float],
            conf_level: float = common.DEFAULT_CONFIDENCE_LEVEL
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

        estimates_tf = [
            tf.constant([estimates[n]]) for n in self.estimate_names
        ]
        params_tf = [
            tf.constant([params[n]]) for n in self.param_names_in_sim_order
        ]

        estimates_uniform = self._estimates_to_net(*estimates_tf)
        params_uniform = self._params_to_net(*params_tf)

        known_params = self.cinet.known_params(params_uniform)
        p = self.pnet.p(estimates_uniform, params_uniform)
        lower_transformed, upper_transformed = self.cinet.ci(
            estimates_uniform,
            known_params,
            tf.constant([1. - conf_level])
        )

        # TODO: improve this: it should be explicitly pulling the estimate(s)
        #       rather than relying on it/them being first in the
        #       sim_to_net_order.
        index_of_estimate = self.sim_to_net_order[0]

        lower = self._params_from_net(lower_transformed)[index_of_estimate]
        upper = self._params_from_net(upper_transformed)[index_of_estimate]

        # TODO: currently only makes an interval for the first estimate
        estimate_name = self.estimate_names[0]
        lower_name = estimate_name + "_lower"
        upper_name = estimate_name + "_upper"

        return {
            "p": self._tensor1_first_elem_to_float(p),
            lower_name: self._tensor1_first_elem_to_float(lower),
            upper_name: self._tensor1_first_elem_to_float(upper)
        }

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
            params_transformed: Tensor2[tf32, Samples, Params]
    ) -> Tensor2[tf32, Samples, Estimates]:

        params = self._params_from_net(params_transformed)
        estimates = self.sampling_distribution_fn(*params).values()
        estimates_transformed = self._estimates_to_net(*estimates)

        return estimates_transformed

    @tf.function
    def _contrast_fn_net_interface(
            self,
            params_transformed: Tensor2[tf32, Samples, Params]
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
            transformed: Tensor2[tf32, Samples, Params]
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
            transformed: Tensor2[tf32, Samples, Params]
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
            *params: Tensor1[tf32, Samples]
    ) -> Tensor2[tf32, Samples, Params]:

        dist_par = zip(self.param_dists_in_sim_order, params)
        unif_sim_order = [dist.to_std_uniform(par) for dist, par in dist_par]
        unif_net_order = self._reorder(unif_sim_order, self.sim_to_net_order)
        params_std_uniform_stacked = tf.stack(unif_net_order, axis=1)
        return self._std_uniform_to_net(params_std_uniform_stacked)

    @tf.function
    def _estimates_to_net(
            self,
            *estimates: Tensor1[tf32, Samples]
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

    def _tensor1_first_elem_to_float(
            self,
            tensor: Tensor1[tf32, Samples]
    ) -> float:

        return float(tensor.numpy()[0])

    @tf.function
    def _std_uniform_to_net(
            self,
            std_uniform: Tensor2[tf32, Samples, Union[Estimates, Params]]
    ) -> Tensor2[tf32, Samples, Union[Estimates, Params]]:

        return sampling.uniform_from_std_uniform(
            std_uniform, common.PARAMS_MIN, common.PARAMS_MAX
        )

    @tf.function
    def _std_uniform_from_net(
            self,
            transformed: Tensor2[tf32, Samples, Union[Estimates, Params]]
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
            param_distributions_named: Dict[str, Distribution]
    ) -> List[str]:

        test_params = self._generate_params_test_sample(
            param_distributions_named, 2
        )
        estimates = self.sampling_distribution_fn(*test_params)
        return list(estimates.keys())

    def _generate_params_test_sample(
            self,
            param_distributions_named: Dict[str, Distribution],
            n: int
    ) -> List[Tensor1[tf32, Samples]]:

        sim_params = self._get_tf_params(self.sampling_distribution_fn)
        dists = [param_distributions_named[p] for p in sim_params]
        params = [d.from_std_uniform(tf.random.uniform((n,))) for d in dists]
        return params

    def _get_tf_params(
            self,
            tf_function: TFFunction,
    ) -> List[str]:

        return tf_function.function_spec.arg_names

    def _align_simulation_params(
            self,
            param_distributions_named: dict
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

        assert(
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
            param_distributions_named: dict
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
            common.PARAMS_MAX
        )
        as_params = self._params_from_net(test_params)
        as_uniform = self._params_to_net(*as_params)
        errors = tf.math.abs(as_uniform - test_params)                         # type: ignore

        return tf.math.reduce_max(errors)
