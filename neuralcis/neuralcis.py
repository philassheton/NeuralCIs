from abc import ABC, abstractmethod

import tensorflow as tf
import neuralcis.common as common
from neuralcis._SinglePNet import _SinglePNet
from neuralcis._CINet import _CINet
from neuralcis import sampling

from typing import Tuple, Union
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2
from tensor_annotations.tensorflow import float32 as tf32
from neuralcis.common import Samples, Estimates, Params


class NeuralCIs(ABC):
    """Train neural networks that compute *p*-values and confidence intervals.

    The following instance variables are important to the correct setup of
    the model:

    :ivar self.num_good_param: An int, the number of parameters to be estimated.
        In Version 1.0.0, this defaults to 1 and **CANNOT** be changed.
    :ivar self.num_nuisance_param: An int, the number of nuisance variables that
        need to be removed from the estimation.  In Version 1.0.0, this
        defaults to 0 and **CANNOT** be changed.
    :ivar self.num_known_param: An int, the number of further parameters that are
        known *a priori* (for example sample sizes).  This defaults to 0 and
        **CAN** be changed.

    The following methods must also be implemented:

    :func simulate_sampling_distribution: Generate samples from the sampling
        distribution.
    :func params_from_std_uniform: Mapping from a set of standard uniform
        variables to an appropriate sampling of the parameters.
    :func params_from_std_uniform: The inversion of the mapping in `
        params_from_std_uniform`.
    :func estimates_to_std_uniform: A mapping from parameter estimates into
        a similar space to that used by `params_from_std_uniform`.  In most
        cases this can be just the same transform.

    Once an instance of the new class is instantiated, the following members
    allow the model to be fit, and for p-values and confidence intervals to
    be calculated for a new observation:

    :func fit:  Fit the networks to the simulation.
    :func p_and_ci: Calculate *p*-value and confidence interval for a novel
        observation.
    """

    num_good_param: int = 1
    num_nuisance_param: int = 0
    num_known_param: int = 0

    @abstractmethod
    def simulate_sampling_distribution(
            self,
            *params: Tensor1[tf32, Samples]
    ) -> Tuple[Tensor1[tf32, Samples], ...]:

        """Generate samples from the sampling distribution.

        This must be a `tf.function` and should take as arguments the
         parameters to the model, each as a 1D `Tensor` of `float32s`, and
         return a tuple of parameter estimates, each also being a 1D `Tensor`.
         In Version 1.0.0 this tuple should only contain one Tensor.  The
         *n*th element of each `Tensor` returned should be a single sample
         drawn from the sampling distribution defined by the parameters
         defined at the *n*th element of each input `Tensor`.
         """

    @abstractmethod
    def params_from_std_uniform(
            self,
            *params: Tensor1[tf32, Samples]
    ) -> Tuple[Tensor1[tf32, Samples], ...]:

        """Transform standard uniform random variables into the parameters.

        This must be a `tf.function` and should take as arguments as many 1D
        `tf.float32` `Tensor`s as there are parameters.  These `Tensors` are
        random standard uniform and should be converted into an appropriate
        sampling of the parameter values within the desired ranges.  The
        return value should be a tuple of transformed 1D `Tensor`s.
        """

    @abstractmethod
    def params_to_std_uniform(
            self,
            *args: Tensor1[tf32, Samples],
            **kwargs: Tensor1[tf32, Samples]
    ) -> Tuple[Tensor1[tf32, Samples], ...]:

        """Invert the `params_from_std_uniform` mapping.

        This must be a `tf.function` and should take as arguments the
        parameters to the model, each as a 1D `Tensor` of `tf.float32`s. The
        return value should be a tuple of 1D `Tensor`s of the same cardinality
        as the inputs, which are mapped from the inputs according to the exact
        inverse of the `params_from_std_uniform` mapping.
        """

    @abstractmethod
    def estimates_to_std_uniform(
            self,
            *args: Tensor1[tf32, Samples],
            **kwargs: Tensor1[tf32, Samples]
    ) -> Tuple[Tensor1[tf32, Samples], ...]:

        """Transform the *estimates* similarly to `params_to_std_uniform`.

        This must be a `tf.function` and should take as arguments the
        estimates sampled by the model, each as a 1D `Tensor` of `
        tf.float32s` (in Version 1.0.0, this is just a single input).
        The return value is a tuple of 1D `Tensor`s (again, with just one
        element).  This should apply a similar mapping to that applied to
        the parameters in `params_to_std_uniform`; it is not necessary for
        it to result in a standard uniform variable (unlike with the
        parameters) but it should be *roughly* distributed between 0 and 1.
        In most cases, the same transform can be used for the estimates as
        for the parameters; care must only be taken that it does not generate
        significant outliers far beyond 0 or 1.
        """

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
            *estimates_and_params: float,
            conf_level: float = common.DEFAULT_CONFIDENCE_LEVEL
    ) -> Tuple[float, float, float]:

        """Calculate the p-value and confidence interval for a novel case.

        This is the "user-friendly" interface to the network.  Pass in a
        single estimate, null parameter value, and known parameters, and
        it will return p-value, lower bound, upper bound.

        :param estimate, null param value, known params: Pass in one float
            value for each of the following (*positional params*):
            (1) observed parameter estimate, (2) null parameter value for
            *p*-value and (3, 4, ...) known parameter values.
        :param conf_level: A float (default .95).  Confidence level for the
            confidence interval.
        :return: Tuple of three floats: p-value, lower and upper CI bounds.
        """

        estimates_and_params_expanded = [
            tf.constant([x]) for x in estimates_and_params
        ]
        estimates = estimates_and_params_expanded[0:self.num_good_param]
        params = estimates_and_params_expanded[self.num_good_param:]

        estimates_uniform = self._estimates_to_transformed(*estimates)
        params_uniform = self._params_to_transformed(*params)

        known_params = self.cinet.known_params(params_uniform)
        p = self.pnet.p(estimates_uniform, params_uniform)
        lower_transformed, upper_transformed = self.cinet.ci(
            estimates_uniform,
            known_params,
            tf.constant([1. - conf_level])
        )

        lower, *_ = self._params_from_transformed(lower_transformed)
        upper, *_ = self._params_from_transformed(upper_transformed)

        return (
            self._tensor1_first_elem_to_float(p),
            self._tensor1_first_elem_to_float(lower),
            self._tensor1_first_elem_to_float(upper)
        )

    ###########################################################################
    #
    #  Private members
    #
    #   -- only single underscores because Tensorflow does not support double
    #       underscore private functions.
    #
    ###########################################################################

    def __init__(self) -> None:
        assert(self.num_good_param == 1)
        assert(self.num_nuisance_param == 0)

        assert(self._max_error_of_reverse_mapping().numpy() <
               common.ERROR_ALLOWED_FOR_PARAM_MAPPINGS)

        self.pnet = _SinglePNet(self._sampling_distribution_fn,
                                self.num_known_param)

        self.cinet = _CINet(self.pnet,
                            self._sampling_distribution_fn,
                            self.num_known_param)

    @tf.function
    def _sampling_distribution_fn(
            self,
            params_transformed: Tensor2[tf32, Samples, Params]
    ) -> Tensor2[tf32, Samples, Estimates]:

        params = self._params_from_transformed(params_transformed)
        estimates = self.simulate_sampling_distribution(*params)
        estimates_transformed = self._estimates_to_transformed(*estimates)

        return estimates_transformed

    def _get_num_params(self) -> int:
        return (
                self.num_good_param +
                self.num_nuisance_param +
                self.num_known_param
        )

    @tf.function
    def _std_uniform_to_transformed(
            self,
            std_uniform: Tensor2[tf32, Samples, Union[Estimates, Params]]
    ) -> Tensor2[tf32, Samples, Union[Estimates, Params]]:

        return sampling.uniform_from_std_uniform(
            std_uniform, common.PARAMS_MIN, common.PARAMS_MAX
        )

    @tf.function
    def _std_uniform_from_transformed(
            self,
            transformed: Tensor2[tf32, Samples, Union[Estimates, Params]]
    ) -> Tensor2[tf32, Samples, Union[Estimates, Params]]:

        return sampling.uniform_to_std_uniform(
            transformed, common.PARAMS_MIN, common.PARAMS_MAX
        )

    @tf.function
    def _estimates_to_transformed(
            self,
            *estimates: Tensor1[tf32, Samples]
    ) -> Tensor2[tf32, Samples, Estimates]:

        estimates_std_uniform_split = self.estimates_to_std_uniform(*estimates)
        estimates_std_uniform = tf.stack(estimates_std_uniform_split, axis=1)
        return self._std_uniform_to_transformed(estimates_std_uniform)

    @tf.function
    def _params_to_transformed(
            self,
            *params: Tensor1[tf32, Samples]
    ) -> Tensor2[tf32, Samples, Params]:

        params_std_uniform_split = self.params_to_std_uniform(*params)
        params_std_uniform = tf.stack(params_std_uniform_split, axis=1)
        return self._std_uniform_to_transformed(params_std_uniform)

    @tf.function
    def _params_from_transformed(
            self,
            transformed: Tensor2[tf32, Samples, Params]
    ) -> Tuple[Tensor1[tf32, Samples], ...]:

        std_uniform = self._std_uniform_from_transformed(transformed)
        std_uniform_split = tf.unstack(std_uniform,
                                       num=self._get_num_params(),
                                       axis=1)

        return self.params_from_std_uniform(*std_uniform_split)

    def _max_error_of_reverse_mapping(self) -> Tensor0[tf32]:
        """Test how accurately the reverse mapping can construct the forward
        mapping.

        :return: Float32 Tensor, maximum absolute error.
        """
        test_params = tf.random.uniform(
            (common.SAMPLES_TO_TEST_PARAM_MAPPINGS, self._get_num_params()),
            common.PARAMS_MIN,
            common.PARAMS_MAX
        )
        as_params = self._params_from_transformed(test_params)
        as_uniform = self._params_to_transformed(*as_params)
        errors = tf.math.abs(as_uniform - test_params)                         # type: ignore

        return tf.math.reduce_max(errors)

    def _tensor1_first_elem_to_float(self, tensor: Tensor1[tf32, Samples]) -> float:
        return float(tensor.numpy()[0])
