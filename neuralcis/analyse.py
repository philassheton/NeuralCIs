import tensorflow as tf
import numpy as np
from tqdm import tqdm

# Use matplotlib where possible
import matplotlib
import matplotlib.pyplot as plt

# Use plotly when needed for advanced features
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neuralcis import NeuralCIs
from neuralcis.distributions import Distribution

from typing import Sequence, Dict, Optional, Callable


def __param_names_and_medians(
        cis: NeuralCIs,
        **value_overrides: float,
) -> Dict[str, float]:

    param_values = value_overrides
    for name, dist in zip(cis.param_names_in_sim_order,
                          cis.param_dists_in_sim_order):
        if name not in param_values:
            param_values[name] = dist.from_std_uniform(0.5).numpy()

    return param_values


def __param_names_and_random_values(
        cis: NeuralCIs,
        **value_overrides: float,
) -> Dict[str, float]:

    param_values = value_overrides
    for name, dist in zip(cis.param_names_in_sim_order,
                          cis.param_dists_in_sim_order):
        if name not in param_values:
            param_values[name] = dist.from_std_uniform(np.random.rand())

    return param_values


def __param_names_and_distributions(
        cis: NeuralCIs,
) -> Dict[str, Distribution]:

    return {name: dist for name, dist in zip(cis.param_names_in_sim_order,
                                             cis.param_dists_in_sim_order)}


def __repeat_params_np(
        num_points: int,
        **param_floats,
) -> Dict[str, np.ndarray]:

    params = {}
    for name, value in param_floats.items():
        params[name] = np.repeat(value, num_points)
    return params


def __repeat_params_tf(
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


def __plot_p_value_distribution_once(
        cis: NeuralCIs,
        fig: plotly.graph_objs._figure.Figure,
        row: int,
        col: int,
        num_samples: int = 10000,
        randomize_unspecified_params: bool = True,
        **param_values: float,
) -> None:

    if randomize_unspecified_params:
        param_values = __param_names_and_random_values(cis, **param_values)
    else:
        param_values = __param_names_and_medians(cis, **param_values)
    param_tensors = __repeat_params_tf(num_samples, **param_values)
    param_arrays = __repeat_params_np(num_samples, **param_values)
    estimates = cis.sampling_distribution_fn(**param_tensors)
    ps = cis.ps_and_cis(estimates, param_arrays)
    title = __summarize_values(**param_values)
    fig.add_trace(
        go.Histogram(x=ps["p"], hovertext=title),
        row=row + 1,
        col=col + 1,
    )


def __get_axis_types(cis: NeuralCIs) -> Dict[str, str]:
    return {name: dist.axis_type for name, dist in zip(
                        cis.param_names_in_sim_order,
                        cis.param_dists_in_sim_order)}


# Because set_xscale does not work in 3d (grrr)
def __plot_3d_with_axis_types(
        plot_fn: Callable,  # e.g. ax.plot_surface
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        x_axis_type: str,   # currently either "log" or "linear"
        y_axis_type: str,
):

    def round_log_scale_ticks(v: np.ndarray, sf=1):
        orders_of_magnitude = np.floor(np.log10(v))
        rounding_factor = 10 ** (sf - orders_of_magnitude - 1)
        rounded = np.round(v * rounding_factor) / rounding_factor
        return rounded

    def transform_axis(v: np.ndarray, axis_type: str):
        if axis_type == "linear":
            locator = matplotlib.ticker.AutoLocator()
            tick_labels = locator.tick_values(v.min(), v.max())
            tick_points = tick_labels
        elif axis_type == "log":
            v = np.log(v)
            tick_labels = np.exp(np.linspace(v.min(), v.max(), 5))             # TODO: This approach produced more pleasing results than matplotlib.ticker.LogLocator, but will produce odd results if we ever want a log parameter that starts high and only goes up a bit.
            tick_labels = round_log_scale_ticks(tick_labels)
            tick_points = np.log(tick_labels)
        else:
            raise Exception(("Currently can only handle log and linear "
                             "axis_type"))

        return v, tick_points, tick_labels

    x, x_ticks, x_tick_labels = transform_axis(x, x_axis_type)
    y, y_ticks, y_tick_labels = transform_axis(y, y_axis_type)

    plot_fn(x, y, z)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)


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

    param_overrides = __param_names_and_medians(cis, **param_overrides)
    params_tensors = __repeat_params_tf(num_points, **param_overrides)

    estimates = cis.sampling_distribution_fn(**params_tensors)
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

    param_values = __param_names_and_medians(cis, **param_values)
    param_defaults = __repeat_params_tf(num_points, **param_values)
    param_samples = cis._params_from_net(cis.pnet.sample_params(num_points))
    param_samples = {name: value for name, value in
                     zip(cis.param_names_in_sim_order, param_samples)}
    axis_types = __get_axis_types(cis)

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


def plot_p_value_distributions(
        cis: NeuralCIs,
        num_rows: int = 10,
        num_cols: int = 20,
        num_samples: int = 10000,
        randomize_unspecified_params: bool = True,
        **param_values: float,
) -> None:

    """Plot distribution of p-values at randomly selected parameter values.

    Will generate p-value distributions for `num_rows * num_cols` different
    randomly generated parameter values, and displays these as a grid of
    histograms.  Individual parameters can also be overridden by passing
    their fixed value as a float to the function, and/or all parameters
    can also be set to their median values by setting
    `randomize_unspecified_params=False`.

    :param cis:  A NeuralCIs object to interrogate.
    :param num_rows:  An int, default 10; number of rows in the histogram grid.
    :param num_cols:  An int, default 20; number of cols in the histogram grid.
    :param num_samples:  An int, default 10000; number of samples per dist'n.
    :param randomize_unspecified_params:  A bool, default True; if set to True,
        then any param values not overridden in **param_values will be chosen
        randomly according to the param sampling distribution.  If set to
        False, the median will be used instead.
    :param **param_values:  A set of float overrides used to set a given
        parameter to a given fixed value in all histograms.

    Examples:

        plot_p_value_distributions(cis)
        plot_p_value_distributions(cis, n=20.)
    """

    print("Generating plots; this may take a few seconds")
    fig = make_subplots(num_rows, num_cols)
    for row in tqdm(range(num_rows)):
        for col in range(num_cols):
            __plot_p_value_distribution_once(cis,
                                             fig, row, col,
                                             num_samples,
                                             randomize_unspecified_params,
                                             **param_values)
    title = ("P-values distribution at different parameter values.  "
             "(Hover over a graph to see the parameter values)")
    fig.update_layout(showlegend=False, title=title)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    print("Done!  Showing graph.")
    fig.show()


def cis_surface(
        cis: NeuralCIs,
        x_name: str,
        y_name: str,
        z_name: str = "p",
        x_is_param: bool = False,
        y_is_param: bool = False,
        param_overrides: Optional[Dict] = None,
        estimate_overrides: Optional[Dict] = None,
        num_grid: int = 100,
) -> None:

    """Plot surface of p, z0, z1, etc against any two other variables.

    :param cis:  A NeuralCIs object to plot.
    :param x_name:  A str, name of the variable to be plotted on x-axis.
    :param y_name:  A str, name of the variable to be plotted on y-axis.
    :param z_name:  A str, default value "p", name of variable to be on
        z-axis.  This can currently be any of "p" or "z0", "z1", ..., or
        "feeler_log_vol", "feeler_p_intersect"
    :param x_is_param:  A bool, default False.  Set to True if the x_name
        should be interpreted as the name of a param, rather than an estimate.
    :param y_is_param:  A bool, default False.  See x_is_param.
    :param param_overrides:  An optional dict of "other" params (i.e. not
        x or y) which should NOT be set to their median values but rather to
        the values in this dict.
    :param estimate_overrides:  An optional dict; see param_overrides.
    :param num_grid:  An int, default 100; number of grid points on each axis.

    Examples

        cis_surface(cis, "mu", "sigma")
        cis_surface(cis, "mu", "mu", x_is_param=True)
        cis_surface(cis, "mu", "sigma", "z0")
        cis_surface(cis, "mu", "sigma", param_overrides={"n": 10.})
    """

    if param_overrides is None:
        param_values = __param_names_and_medians(cis)
    else:
        param_values = __param_names_and_medians(cis, **param_overrides)

    axis_types = __get_axis_types(cis)

    estimate_values = {name: param_values[name] for name in cis.estimate_names}
    if estimate_overrides is not None:
        for name, override in estimate_overrides.items():
            estimate_values[name] = override

    param_dists = __param_names_and_distributions(cis)
    std_unif_linspace = tf.constant(np.linspace(0., 1., num_grid), tf.float32)
    x_linspace = param_dists[x_name].from_std_uniform(std_unif_linspace)
    y_linspace = param_dists[y_name].from_std_uniform(std_unif_linspace)

    if x_is_param:
        param_values[x_name] = x_linspace
    else:
        estimate_values[x_name] = x_linspace

    if y_is_param:
        param_values[y_name] = y_linspace
    else:
        estimate_values[y_name] = y_linspace

    xs, ys, zs = cis.values_grid(estimate_values, param_values, (z_name,))

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    __plot_3d_with_axis_types(ax.plot_surface, ax,
                              xs, ys, zs,
                              axis_types[x_name], axis_types[y_name])
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)
    fig.show()
