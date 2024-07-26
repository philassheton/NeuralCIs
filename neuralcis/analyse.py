import tensorflow as tf
import matplotlib.pyplot as plt

from neuralcis import NeuralCIs

from typing import Sequence, Dict, Optional


def __param_names_and_defaults(
        cis: NeuralCIs,
        **default_overrides: float,
) -> Dict[str, float]:

    param_values = default_overrides
    for name, dist in zip(cis.param_names_in_sim_order,
                          cis.param_dists_in_sim_order):
        if name not in param_values:
            param_values[name] = dist.from_std_uniform(0.5).numpy()

    return param_values


def __repeat_params(
        num_points: int,
        **param_floats,
) -> Dict[str, tf.Tensor]:

    params = {}
    for name, value in param_floats.items():
        params[name] = tf.repeat(value, num_points)
    return params


def __summarize_values(
        preamble: str = "",
        **values,
) -> str:

    title = preamble
    spacer = ""
    for name, value in values.items():
        title += f"{spacer}{name}={value:.2f}"
        spacer = "; "
    title += "."

    return title


def visualize_sampling_fn(
        cis: NeuralCIs,
        num_points: int = 100,
        **param_overrides,
) -> None:

    """Visualize the sampling and contrast functions at a given set of param
    values.  Estimate variables are plotted pair by pair.  If 
    a given variable is not provided in param_overrides, the median of the
    distribution for that param will be used.

    Examples:

        visualize_sampling_fn(cis)
        visualize_sampling_fn(cis, mu=0., n=4.)
        visualize_sampling_fn(cis, mu=0., n=4., num_points=10000)
    """

    param_overrides = __param_names_and_defaults(cis, **param_overrides)
    params = __repeat_params(num_points, **param_overrides)

    estimates = cis.sampling_distribution_fn(**params)
    num_estimates = len(estimates)
    fig, ax = plt.subplots(num_estimates, num_estimates)
    for i_x, name_x in enumerate(estimates):
        for i_y, name_y in enumerate(estimates):
            ax[i_y][i_x].scatter(estimates[name_x], estimates[name_y])
            if i_y == num_estimates - 1:
                ax[i_y][i_x].set_xlabel(name_x)
            if i_x == 0:
                ax[i_y][i_x].set_ylabel(name_y)

    title = __summarize_values("Dist'n of estimates with params set at: ",
                               **param_overrides)
    fig.suptitle(title, wrap=True)
    fig.show()


def plot_params_vs_estimates(
        cis: NeuralCIs,
        num_points: int = 1000,
        param_names: Optional[Sequence[str]] = None,
        estimate_names: Optional[Sequence[str]] = None,
        **param_values,
) -> None:

    """Plot each param vs the distribution of the estimates as it is varied.

    In each plot, only the param labelled on the x-axis is varied, while all
    other params are fixed at either the value provided in **param_values, or
    else at the median of its sampling distribution.  Unless specified via
    param_names and estimate_names, all params will be plotted vs all
    estimates.

    :param cis: A NeuralCIs object to analyse.
    :param num_points:  int (default 1000); number of samples per mini-plot.
    :param param_names:   Optional list of strs giving the names of params to
        be plotted, in the order they are to be plotted.  If unspecified, all
        params from the model will be plotted in the order in which they
        appear in the signature of the sampling function.
    :param estimate_names:  Optional list of strs giving the names of estimates
        to be plotted, in the order they are to be plotted.  If unspecified,
        all estimates returned by the sampling function will be plotted.
    :param param_values:  Optional floats, overrides for default values of
        the params, to be used whenever that param is not part of the plot.
        If not specified, these will be taken to be the median of their
        respective sampling distributions.

    Examples:

        plot_params_vs_estimates(cis)
        plot_params_vs_estimates(cis,
                                 param_names=("mu1", "mu2"),
                                 estimate_names=("mu1", "mu2"),
                                 mu1=0.,
                                 mu2=0.)
    """

    if param_names is None:
        param_names = cis.param_names_in_sim_order
    if estimate_names is None:
        estimate_names = cis.estimate_names

    param_values = __param_names_and_defaults(cis, **param_values)
    param_defaults = __repeat_params(num_points, **param_values)
    param_samples = cis._params_from_net(cis.pnet.sample_params(num_points))
    param_samples = {name: value for name, value in
                     zip(cis.param_names_in_sim_order, param_samples)}
    axis_types = {name: dist.axis_type for name, dist in zip(
                                                cis.param_names_in_sim_order,
                                                cis.param_dists_in_sim_order)}

    fig, ax = plt.subplots(len(estimate_names), len(param_names))
    for col, param in enumerate(param_names):
        this_params = param_defaults | {param: param_samples[param]}
        this_estimates = cis.sampling_distribution_fn(**this_params)
        for row, estimate in enumerate(estimate_names):
            ax[row][col].scatter(this_params[param], this_estimates[estimate])
            ax[row][col].set_xscale(axis_types[param])
            ax[row][col].set_yscale(axis_types[estimate])
            if row == len(estimate_names) - 1:
                ax[row][col].set_xlabel(param)
            if col == 0:
                ax[row][col].set_ylabel(estimate)

    title = __summarize_values("Params vs estimates, with default params: ",
                               **param_values)
    fig.suptitle(title, wrap=True)
    fig.show()
