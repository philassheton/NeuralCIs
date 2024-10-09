import copy
from collections import abc

import pandas as pd
import tensorflow as tf
import numpy as np
from scipy import stats

from tensorflow.python.eager.def_function import Function as TFFunction        # type: ignore
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
from neuralcis.common import HAT

from typing import Sequence, Dict, Optional, Callable, Tuple, Union, Any
from tensor_annotations.tensorflow import Tensor1, float32 as tf32
from neuralcis.common import Samples


def __param_names_and_medians(
        cis: NeuralCIs,
        **value_overrides: float,
) -> Dict[str, float]:

    param_values = {n: value_overrides[n]
                    for n in cis.param_names_in_sim_order
                    if n in value_overrides}
    for name, dist in zip(cis.param_names_in_sim_order,
                          cis.param_dists_in_sim_order):
        if name not in param_values:
            param_values[name] = dist.from_std_uniform(0.5).numpy()

    return param_values


def __param_names_and_random_values(
        cis: NeuralCIs,
        from_estimates_box_only: bool = False,
        **value_overrides,
) -> Dict[str, float]:

    if len(value_overrides) and not from_estimates_box_only:
        raise ValueError('If drawing parameters randomly from the '
                         'param sampling distribution, it is only '
                         'possible to draw all parameters at once, '
                         'since I do not currently have a way to '
                         'compute a conditional distribution on this '
                         'at present.  You can still randomize '
                         'parameters within the estimates box AND '
                         'have parameter overrides, by setting '
                         'from_estimates_box_only=True.  This '
                         'will only generate params within the valid '
                         'estimates box, but will allow you to fix '
                         'any number of the parameters.')

    if from_estimates_box_only:
        param_values = value_overrides
        for name, dist in zip(cis.param_names_in_sim_order,
                              cis.param_dists_in_sim_order):
            if name not in param_values:
                r = np.random.rand()
                param_values[name] = dist.from_std_uniform_valid_estimates(r)
    else:
        random_params = cis.sample_params(1)
        param_values = {n: float(p.numpy()) for n, p in random_params.items()}

    return param_values


def __add_estimates_equal_to_params(
        cis: NeuralCIs,
        param_values: Dict[str, Any],
        **value_overrides,
) -> Dict[str, Any]:

    estimate_values = {name: param_values[dehat]
                       for name, dehat in zip(cis.estimate_names,
                                              cis._estimate_names_dehatted())}
    for estimate_name in estimate_values.keys():
        if estimate_name in value_overrides:
            estimate_values[estimate_name] = value_overrides[estimate_name]

    return estimate_values | param_values


def __estimate_and_param_names_and_distributions(
        cis: NeuralCIs,
) -> Dict[str, Distribution]:

    param_dists = {name: dist
                   for name, dist in zip(cis.param_names_in_sim_order,
                                         cis.param_dists_in_sim_order)}
    return __add_estimates_equal_to_params(cis, param_dists)


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
        params[name] = tf.cast(tf.repeat(value, num_points), tf.float32)
    return params


def __axis_linspace(
        cis: NeuralCIs,
        name: str,
        lims: Optional[Sequence[float]] = None,                                # If None, use valid estimates range
        num_grid: int = 100,
):
    std_unif_linspace = tf.constant(np.linspace(0., 1., num_grid), tf.float32)
    param_dists = __estimate_and_param_names_and_distributions(cis)
    if lims is None:
        u = std_unif_linspace
        linspace = param_dists[name].from_std_uniform_valid_estimates(u)
    else:
        lims_uniform = param_dists[name].to_std_uniform(tf.constant(lims))
        low_uniform = lims_uniform[0]
        width_uniform = lims_uniform[1] - lims_uniform[0]
        linspace_uniform = std_unif_linspace * width_uniform + low_uniform
        linspace = param_dists[name].from_std_uniform(linspace_uniform)

    return linspace


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


def __get_one_p_value_distribution(
        cis: NeuralCIs,
        num_samples: int,
        randomize_unspecified_params: bool,
        from_estimates_box_only: bool,
        **param_values: float,
) -> Dict[str, Union[np.ndarray, float]]:

    if randomize_unspecified_params:
        param_values = __param_names_and_random_values(cis,
                                                       from_estimates_box_only,
                                                       **param_values)
    else:
        param_values = __param_names_and_medians(cis, **param_values)

    param_tensors = __repeat_params_tf(num_samples, **param_values)
    estimates = cis.sampling_distribution_fn(**param_tensors)
    ps_and_cis = cis.ps_and_cis(**(estimates | param_tensors))
    ps = ps_and_cis["p"]

    return {"p": ps} | param_values


def __plot_p_value_distribution_once(
        cis: NeuralCIs,
        fig: plotly.graph_objs.Figure,
        row: int,
        col: int,
        num_samples: int,
        randomize_unspecified_params: bool,
        from_estimates_box_only: bool,
        **param_values: float,
) -> None:

    ps = __get_one_p_value_distribution(cis, num_samples,
                                        randomize_unspecified_params,
                                        from_estimates_box_only,
                                        **param_values)["p"]
    title = __summarize_values(**param_values)
    fig.add_trace(
        go.Histogram(x=ps, hovertext=title),
        row=row + 1,
        col=col + 1,
    )


def __get_axis_types(cis: NeuralCIs) -> Dict[str, str]:
    param_axis_types = {name: dist.axis_type
                        for name, dist in zip(cis.param_names_in_sim_order,
                                              cis.param_dists_in_sim_order)}
    return __add_estimates_equal_to_params(cis, param_axis_types)


# Because set_xscale does not work in 3d (grrr)
def __plot_3d_with_axis_types(
        plot_fn: Callable,  # e.g. ax.plot_surface
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        x_axis_type: str,   # currently either "log" or "linear"
        y_axis_type: str,
        z_axis_lims: Optional[Sequence[float]],
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

    if z_axis_lims is not None:
        z = tf.math.maximum(z, z_axis_lims[0])
        z = tf.math.minimum(z, z_axis_lims[1])

    plot_fn(x, y, z)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)


def __make_pandas(
        estimates_and_params: Dict,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        num_rows_to_print: int = 100,
        **further_measures: Union[Tensor1[tf32, Samples], np.ndarray],
) -> pd.DataFrame:

    all_dfs = []
    if estimates_and_params is not None:
        all_dfs.append(pd.DataFrame(estimates_and_params))
    if len(further_measures) > 0:
        all_dfs.append(pd.DataFrame(further_measures))

    df = pd.concat(all_dfs, axis=1)
    if sort_by is not None:
        df = df.sort_values(by=sort_by,
                            ascending=ascending).reset_index().drop("index",
                                                                    axis=1)

    if num_rows_to_print > 0:
        with pd.option_context("display.max_columns", None,
                               "display.max_rows", None,
                               "display.float_format", "{:.3f}".format):
            print(df.head(num_rows_to_print))

    return df


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

    # TODO: With new sampling approach, some of these will be out of range
    #       at the default values.
    param_samples = cis.sample_params(num_points)
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


def plot_p_value_cdfs(
        cis: NeuralCIs,
        num_cdfs: int = 200,
        num_samples: int = 10000,
        randomize_unspecified_params: bool = True,
        from_estimates_box_only: bool = True,
        sampling_dist_percent: float = 99,
        params_df: Optional[pd.DataFrame] = None,
        params_df_rows: Union[None, int, Sequence[int]] = None,
        **param_values: float,
) -> pd.DataFrame:

    """Overlay CDFs of p-values at randomly selected parameter values.

    Will generate p-value distributions for `num_cdfs` different
    randomly generated parameter values, and overlay these CDFs on top of
    each other on a single plot.  Individual parameters can also be overridden
    by passing their fixed value as a float to the function, and/or all
    parameters can also be set to their median values by setting
    `randomize_unspecified_params=False`.

    :param cis:  A NeuralCIs object to interrogate.
    :param num_cdfs:  An int, default 200; number of CDFs in the plot.
    :param num_samples:  An int, default 10000; number of samples per dist'n.
    :param randomize_unspecified_params:  A bool, default True; if set to True,
        then any param values not overridden in **param_values will be chosen
        randomly according to the param sampling distribution.  If set to
        False, the median will be used instead.
    :param from_estimates_box_only:  A bool, default True; constrain randomly
        selected parameters to be inside the "valid estimates" box, to
        avoid testing points that are on the very boundary.  If this is set to
        False, then params are drawn from the full sampling distribution of
        parameters and **param_values must be empty, as it is not at this time
        possible to model conditional distributions within this distribution.
    :param sampling_dist_percent:  A float, default 99; in this case two
        dashed lines will enclose the theoretical inner 99% of the sampling
        distribution of these empirical CDFs, on the null that they come from
        a uniform distribution.
    :param params_df: An optional pandas.DataFrame that contains the param
        values to use, each in their own column.  One example that would be
        valid here is the dataframe returned by a previous run of this
        function.
    :param params_df_rows: An optional int, or sequence of ints, specifying
        which rows of the dataframe to focus on.  If None and a `params_df`
        is passed in, all rows of the DataFrame will be used; if None and no
        `params_df` is passed in, `num_cdf` random parameters will be used.
        If `params_df` is None, this must also be None.
    :param **param_values:  A set of float overrides used to set a given
        parameter to a given fixed value in all histograms.

    Examples:

        plot_p_value_distributions(cis)
        plot_p_value_distributions(cis, n=20.)
    """

    if params_df is None:
        assert params_df_rows is None
        indices = range(num_cdfs)
    else:
        num_df_rows, _ = params_df.shape
        if params_df_rows is None:
            indices = range(num_df_rows)
        elif isinstance(params_df_rows, int):
            indices = [params_df_rows]
        elif isinstance(params_df_rows, abc.Sequence):
            indices = params_df_rows
        else:
            raise Exception("params_df_rows can only be None, int or"
                            " Sequence[int]!!!")

    print("Generating CDFs; this may take a few seconds")
    alpha = 1. / np.sqrt(len(indices))
    fig, axes = plt.subplots(1, 3)
    y = np.linspace(0., 1., num_samples)
    params = {n: np.array([]) for n in cis.param_names_in_sim_order}
    ks = np.array([])
    for i in tqdm(indices):
        if params_df is not None:
            params_i = (
                    params_df.loc[i, cis.param_names_in_sim_order].to_dict() |
                    param_values
            )
        else:
            params_i = param_values

        cdf_etc = __get_one_p_value_distribution(cis,
                                                 num_samples,
                                                 randomize_unspecified_params,
                                                 from_estimates_box_only,
                                                 **params_i)
        cdf = cdf_etc.pop("p")
        cdf.sort()
        for ax in axes[0:2]:
            ax.plot(cdf, y, alpha=alpha, c="black")
        axes[2].plot(cdf, y - cdf, alpha=alpha, c="black")

        ks = np.append(ks, np.max(np.abs(cdf - y)))
        for name, value in cdf_etc.items():
            params[name] = np.append(params[name], value)

    for ax in axes[0:2]:
        ax.plot([0, 1], [0, 1], c="red", linestyle="--", label="Uniform")

        ax.set_xlabel("p-Value")
        ax.set_ylabel("CDF")
        ax.set_aspect("equal")
    axes[2].set_xlabel("p-Value")
    axes[2].set_ylabel("Distance from uniform")
    axes[2].set_box_aspect(1)

    axes[1].plot([.01, 1], [0, .99], c="cyan", linestyle="--", label="+/- .01")
    axes[1].plot([0, .99], [.01, 1], c="cyan", linestyle="--")

    x_pop = y
    se = np.sqrt((x_pop * (1. - x_pop)) / num_samples)
    tail_prob = (1 - sampling_dist_percent / 100.) / 2.
    z = stats.norm.ppf(tail_prob)
    lower = y - se*z
    upper = y + se*z
    dist_name = f"{sampling_dist_percent}% expected"
    axes[1].plot(x_pop, lower, c="blue", linestyle="--", label=dist_name)
    axes[1].plot(x_pop, upper, c="blue", linestyle="--")
    axes[1].set_xlim(0., 0.1)
    axes[1].set_ylim(0., 0.1)
    axes[1].legend(loc="upper left")
    fig.show()

    pandas_sorted = __make_pandas(
        estimates_and_params=params,
        ks=ks,
        sort_by="ks",
    )

    return pandas_sorted


def plot_p_value_distributions(
        cis: NeuralCIs,
        num_rows: int = 10,
        num_cols: int = 20,
        num_samples: int = 10000,
        randomize_unspecified_params: bool = True,
        from_estimates_box_only: bool = True,
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
    :param from_estimates_box_only:  A bool, default True; constrain randomly
        selected parameters to be inside the "valid estimates" box, to
        avoid testing points that are on the very boundary.  If this is set to
        False, then params are drawn from the full sampling distribution of
        parameters and **param_values must be empty, as it is not at this time
        possible to model conditional distributions within this distribution.
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
                                             from_estimates_box_only,
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
        x_lims: Optional[Sequence[float]] = None,
        y_lims: Optional[Sequence[float]] = None,
        z_lims: Optional[Sequence[float]] = None,
        num_grid: int = 100,
        **value_overrides,
) -> None:

    """Plot surface of p, z0, z1, etc against any two other variables.

    :param cis:  A NeuralCIs object to plot.
    :param x_name:  A str, name of the variable to be plotted on x-axis.
    :param y_name:  A str, name of the variable to be plotted on y-axis.
    :param z_name:  A str, default value "p", name of variable to be on
        z-axis.  This can currently be any of "p" or "z0", "z1", ..., or
        "feeler", "feeler_log_vol", or "feeler_p_intersect".
    :param x_is_param:  A bool, default False.  Set to True if the x_name
        should be interpreted as the name of a param, rather than an estimate.
    :param y_is_param:  A bool, default False.  See x_is_param.
    :param param_overrides:  An optional dict of "other" params (i.e. not
        x or y) which should NOT be set to their median values but rather to
        the values in this dict.
    :param x_lims:  An optional sequence [low, high] of floats.  Sets the
        limits that the x-axis will be plotted between.  If the axis is a log
        variable, make sure not to include zero or less than zero in this.
    :param y_lims:  See x_lims.
    :param z_lims:  See x_lims; will be used to limit the values of z.
    :param estimate_overrides:  An optional dict; see param_overrides.
    :param num_grid:  An int, default 100; number of grid points on each axis.

    Examples

        cis_surface(cis, "mu", "sigma")
        cis_surface(cis, "mu", "mu", x_is_param=True)
        cis_surface(cis, "mu", "sigma", "z0")
        cis_surface(cis, "mu", "sigma", param_overrides={"n": 10.})
    """

    if value_overrides is None:
        param_values = __param_names_and_medians(cis)
    else:
        param_values = __param_names_and_medians(cis, **value_overrides)

    axis_types = __get_axis_types(cis)

    estimates_and_params = __add_estimates_equal_to_params(cis, param_values,
                                                           **value_overrides)

    x_linspace = __axis_linspace(cis, x_name, x_lims, num_grid)
    y_linspace = __axis_linspace(cis, y_name, y_lims, num_grid)

    estimates_and_params[x_name] = x_linspace
    estimates_and_params[y_name] = y_linspace

    xs, ys, zs = cis.values_grid((z_name,), **estimates_and_params)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    __plot_3d_with_axis_types(ax.plot_surface, ax,
                              xs, ys, zs,
                              axis_types[x_name], axis_types[y_name], z_lims)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)
    fig.show()


def compare_power_at_h1(
        cis: NeuralCIs,
        accurate_p_fn: Callable[
            [Dict[str, Tensor1[tf32, Samples]],    # Dict of estimates
             Tuple[Tensor1[tf32, Samples], ...]],  # **params
            Tensor1[tf32, Samples]                 # p-value
        ],
        h0_params: Dict[str, float],
        num_samples: int = 5000,
        **h1_params_different_from_h0_params: float,
):

    """Compare the """

    if not isinstance(accurate_p_fn, TFFunction):
        accurate_p_fn = tf.function(accurate_p_fn)

    h1_params = copy.deepcopy(h0_params) | h1_params_different_from_h0_params

    h0_params = __repeat_params_tf(num_samples, **h0_params)
    h1_params = __repeat_params_tf(num_samples, **h1_params)
    estimates = cis.sampling_distribution_fn(**h1_params)
    ps_neural = cis.ps_and_cis(**(estimates | h0_params))["p"]
    ps_accurate = accurate_p_fn(**(estimates |h0_params))

    fig, ax = plt.subplots(1, 2)
    ax[0].plot([0, 1], [0, 1], 'r-')
    ax[0].scatter(ps_accurate, ps_neural, alpha=.01)
    ax[0].set_xlabel("Accurate p-Value")
    ax[0].set_ylabel("NeuralCIs p-Value")
    ax[1].hist(ps_accurate, np.arange(0., 1.00001, .05),
               label="Accurate test")
    ax[1].hist(ps_neural, np.arange(0., 1.00001, .05), alpha=.5,
               label="NeuralCIs")
    ax[1].set_xlabel("p-Value")
    ax[1].legend(loc="upper right")

    fig.show()

    tf.print(f"Power of traditional approach: {np.mean(ps_accurate < .05)};")
    tf.print(f"Power of NeuralCIs:            {np.mean(ps_neural < .05)}.")


def compare_techniques_within_estimates_box(
        cis: NeuralCIs,
        accurate_p_fn: Callable[
            [Dict[str, Tensor1[tf32, Samples]],  # Dict of estimates
             Tuple[Tensor1[tf32, Samples], ...]],  # **params
            Tensor1[tf32, Samples]  # p-value
        ],
        accurate_p_name: str = "Accurate Method",
        num_tries: int = 1000,
        **h0_params,
) -> pd.DataFrame:

    dists = __estimate_and_param_names_and_distributions(cis)
    rand_unif = lambda: tf.random.uniform((num_tries,))
    estimates_and_params = {n: d.from_std_uniform_valid_estimates(rand_unif())
                            for n, d in dists.items()}
    estimates_and_params |= cis.sample_params(num_tries)
    estimates_and_params |= __repeat_params_tf(num_tries, **h0_params)

    ps_neural = cis.ps_and_cis(**estimates_and_params)["p"]
    ps_accurate = accurate_p_fn(**estimates_and_params)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], c="red", linestyle="--", label="Equal")
    ax.scatter(ps_accurate.numpy(), ps_neural, alpha=0.1)
    ax.set_xlabel(accurate_p_name)
    ax.set_ylabel("Neural CIs p-Value")
    ax.set_aspect("equal")
    fig.show()

    sorted_pandas = __make_pandas(
        estimates_and_params,
        p_neural=ps_neural,
        p_accurate=ps_accurate,
        p_diff=tf.math.abs(ps_neural - ps_accurate),
        sort_by="p_diff",
    )

    return sorted_pandas
