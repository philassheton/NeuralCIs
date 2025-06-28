from .neuralcis import NeuralCIs
from . import analyse

import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import Tuple, Dict, Optional



def test_param_net_without_going_via_fn_interface(
        cis: NeuralCIs,
        num_samples = 100000,
):

    net_samples = cis.param_sampler.sample_params(num_samples)
    importance_ingreds = cis.param_sampler.feeler_net.call_tf(net_samples)
    samples_on_target = tf.reduce_sum(tf.cast(importance_ingreds[:, 1] > -10.,
                                              tf.int64))
    samples_in_inner = tf.reduce_sum(tf.cast(importance_ingreds[:, 2] > -10.,
                                             tf.int64))
    percent_hit = samples_on_target / num_samples * 100
    percent_inner = samples_in_inner / num_samples * 100
    print(f"{samples_on_target} / {num_samples} ({percent_hit:.1f}%) hit rate")
    print(f"{samples_in_inner} / {num_samples} ({percent_inner:.1f}%) inner")

def test_param_samples(
        cis: NeuralCIs,
        x_name: str,
        y_name: str,
        z_name: str,
        num_samples: int = 1000,
        outer: bool = True,
        colour_by_log_vol: bool = False,
        **params_high_low: Dict[str, Tuple[float, float]],
) -> None:

    total_samples_selected = 0
    total_samples_tried = 0
    total_samples_inside_inner = 0
    total_samples_inside_outer = 0
    params = {n: [] for n in cis.param_names_in_net_order}
    progress = tqdm(total=num_samples)
    while total_samples_selected < num_samples:
        total_samples_tried += 100000
        trial_params = cis.sample_params(100000, outer)

        in_range = []
        for param_name, range in params_high_low.items():
            in_range.append(
                (trial_params[param_name] >= range[0]) &
                (trial_params[param_name] <= range[1])
            )

        in_range = tf.reduce_all(tf.stack(in_range, axis=1), axis=1)
        to_select = tf.where(in_range)[:, 0]

        num_to_select = tf.minimum(
            len(to_select),
            num_samples - total_samples_selected,
            tf.float64,
        )
        to_select = to_select[:num_to_select]

        for n in cis.param_names_in_net_order:
            params[n].append(tf.gather(trial_params[n], to_select, axis=0))
        total_samples_selected += num_to_select

        progress.n = total_samples_selected.numpy()
        progress.refresh()
    progress.close()


    params = {n: tf.concat(params[n], axis=0).numpy()
              for n in cis.param_names_in_net_order}
    colours = "black"

    print("%.0f%% of samples were in inner and %.0f%% in outer" %
          (total_samples_inside_inner / total_samples_tried * 100,
           total_samples_inside_outer / total_samples_tried * 100))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    axis_types = analyse.__get_axis_types(cis)
    scatter = analyse.__plot_3d_with_axis_types(
        ax.scatter3D, ax,
        params[x_name], params[y_name], params[z_name],
        axis_types[x_name], axis_types[y_name], axis_types[z_name],
        x_name, y_name, z_name,
        c=colours,
    )
    if colour_by_log_vol:
        fig.colorbar(scatter, ax=ax, label="ln(Vol)")
    fig.show()

def plot_generator_samples(
        cis: NeuralCIs,
        x_name: Optional[str] = None,
        y_name: Optional[str] = None,
        z_name: Optional[str] = None,
        x_lims: Optional[Tuple[float, float]] = None,
        y_lims: Optional[Tuple[float, float]] = None,
        z_lims: Optional[Tuple[float, float]] = None,
        max_samples: Optional[int] = None,
        valid_only: bool = True,
        inner_only: bool = False,
        colour_by_log_vol: bool = False,
        **param_limits,
) -> None:

    if x_name is None:
        assert y_name is None and z_name is None
        assert cis.num_param() >= 3
        x_name, y_name, z_name = cis.param_names_in_net_order[0:3]

    if x_lims is not None:
        param_limits[x_name] = x_lims
    if y_lims is not None:
        param_limits[y_name] = y_lims
    if z_lims is not None:
        param_limits[z_name] = z_lims

    if inner_only:
        generator = cis.param_sampler.inner_data_generator
    else:
        generator = cis.param_sampler.outer_data_generator
    params = generator.sampled_params
    names = cis.param_names_in_net_order
    if len(param_limits):
        dists = cis.param_dists_in_net_order
        for n, d in zip(names, dists):
            if n not in param_limits:
                param_limits[n] = d.from_std_uniform((-float("inf"),
                                                       float("inf")))
        limits_human = [tf.constant(param_limits[n]) for n in names]
        limits_net = cis._params_human_to_net(*limits_human)
        above_bottom = params >= limits_net[0:1, :]
        below_top = params < limits_net[1:2, :]
        in_range = above_bottom & below_top
        indices = tf.where(tf.reduce_all(in_range, axis=1))[:, 0]
    else:
        indices = tf.range(params.shape[0])

    valid = generator.is_inside_support_region(generator.sampled_targets)
    is_inside_outer_zone = tf.gather(valid, indices)
    if valid_only:
        indices = tf.boolean_mask(indices, is_inside_outer_zone)
        is_inside_outer_zone = tf.boolean_mask(is_inside_outer_zone,
                                               is_inside_outer_zone)

    if max_samples is not None and len(indices) > max_samples:
        indices = tf.random.shuffle(indices)[:max_samples]
    is_inside_outer_zone = tf.gather(valid, indices)

    to_plot_net = tf.gather(params, indices, axis=0)
    to_plot = cis._params_net_to_human(to_plot_net)
    to_plot = {n: p for n, p in zip(names, to_plot)}
    targets = tf.gather(generator.sampled_targets, indices, axis=0)

    if colour_by_log_vol:
        colours = targets[:, 0].numpy()
        vmin = colours.min()
        vmax = colours.max()
    else:
        colour_case = (1 * tf.cast(is_inside_outer_zone, tf.int32))

        lookup_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant([0, 1]),  # input keys
                values=tf.constant(["gray", "red"])
            ),
            default_value=tf.constant("purple")
        )
        colours = lookup_table.lookup(colour_case)
        colours = colours.numpy().astype(str)
        vmin = None
        vmax = None

    axis_types = analyse.__get_axis_types(cis)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    scatter = analyse.__plot_3d_with_axis_types(ax.scatter3D,
                                      ax,
                                      to_plot[x_name].numpy(),
                                      to_plot[y_name].numpy(),
                                      to_plot[z_name].numpy(),
                                      axis_types[x_name],
                                      axis_types[y_name],
                                      axis_types[z_name],
                                      x_name,
                                      y_name,
                                      z_name,
                                      c=colours,
                                      vmin=vmin,
                                      vmax=vmax)

    if colour_by_log_vol:
        fig.colorbar(scatter, ax=ax, label="ln(Vol)")
    fig.show()
