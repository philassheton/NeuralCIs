from neuralcis import NeuralCIs, analyse, common

import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import Tuple, Dict, Optional



def test_param_net_without_going_via_fn_interface(
        cis: NeuralCIs,
        num_samples = 100000,
):

    net_samples = cis.param_sampling_net.sample_params(num_samples)
    importance_ingreds = cis.param_sampling_net.feeler_net.call_tf(net_samples)
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
        in_inner_boundary: bool = True,
        in_outer_boundary: bool = False,
        **params_high_low: Dict[str, Tuple[float, float]],
) -> None:

    params = {n: [] for n in cis.param_names_in_sim_order}
    total_samples_selected = 0
    total_samples_tried = 0
    total_samples_inside_inner = 0
    total_samples_inside_outer = 0
    progress = tqdm(total=num_samples)
    while total_samples_selected < num_samples:
        total_samples_tried += 100000
        trial_params = cis.sample_params(100000)


        # PHIL!!  move this into neuralcis
        sim_order_params = [trial_params[n]
                            for n in cis.param_names_in_sim_order]
        net_params = cis._params_to_net(*sim_order_params)


        importance_ingredients = cis.param_sampling_net.feeler_net.call_tf(
            net_params
        )
        is_in_outer = importance_ingredients[:, 1] > -3.
        is_in_inner = importance_ingredients[:, 2] > -3.

        total_samples_inside_inner += tf.reduce_sum(tf.cast(is_in_inner,
                                                            tf.int64)).numpy()
        total_samples_inside_outer += tf.reduce_sum(tf.cast(is_in_outer,
                                                            tf.int64)).numpy()

        in_range = []
        for param_name, range in params_high_low.items():
            in_range.append(
                (trial_params[param_name] >= range[0]) &
                (trial_params[param_name] <= range[1])
            )
        in_range = tf.reduce_all(tf.stack(in_range, axis=1), axis=1)

        to_select = in_range
        if in_inner_boundary:
            to_select = to_select & is_in_inner
        if in_outer_boundary:
            to_select = to_select & is_in_outer
        to_select = tf.where(to_select)[:, 0]

        num_to_select = tf.minimum(
            len(to_select),
            num_samples - total_samples_selected,
            tf.float64,
        )
        to_select = to_select[:num_to_select]

        for n in cis.param_names_in_sim_order:
            params[n].append(tf.gather(trial_params[n], to_select, axis=0))
        total_samples_selected += num_to_select

        progress.n = total_samples_selected.numpy()
        progress.refresh()
    progress.close()


    params = {n: tf.concat(params[n], axis=0).numpy()
              for n in cis.param_names_in_sim_order}

    print("%.0f%% of samples were in inner and %.0f%% in outer" %
          (total_samples_inside_inner / total_samples_tried * 100,
           total_samples_inside_outer / total_samples_tried * 100))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    axis_types = analyse.__get_axis_types(cis)
    analyse.__plot_3d_with_axis_types(
        ax.scatter3D, ax,
        params[x_name], params[y_name], params[z_name],
        axis_types[x_name], axis_types[y_name], axis_types[z_name],
        x_name, y_name, z_name,
    )

def plot_generator_samples(
        cis: NeuralCIs,
        x_name: str,
        y_name: str,
        z_name: str,
        x_lims: Optional[Tuple[float, float]] = None,
        y_lims: Optional[Tuple[float, float]] = None,
        z_lims: Optional[Tuple[float, float]] = None,
        max_samples: Optional[int] = None,
        valid_only: bool = True,
        **param_limits,
) -> None:

    if x_lims is not None:
        param_limits[x_name] = x_lims
    if y_lims is not None:
        param_limits[y_name] = y_lims
    if z_lims is not None:
        param_limits[z_name] = z_lims

    generator = cis.param_sampling_net.feeler_net.feeler_data_generator
    params = generator.sampled_params
    if len(param_limits):
        names = cis.param_names_in_sim_order
        dists = cis.param_dists_in_sim_order
        for n, d in zip(names, dists):
            if n not in param_limits:
                param_limits[n] = d.from_std_uniform((-float("inf"),
                                                       float("inf")))
        limits_in_sim_order = [param_limits[n] for n in names]
        limits_net = cis._params_to_net(*limits_in_sim_order)
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

    if max_samples is not None and len(indices) > max_samples:
        indices = tf.random.shuffle(indices)[:max_samples]

    to_plot_net = tf.gather(params, indices, axis=0)
    to_plot = cis._params_from_net(to_plot_net)
    to_plot = {n: p for n, p in zip(names, to_plot)}
    targets = tf.gather(generator.sampled_targets, indices, axis=0)
    is_inside_inner_zone = targets[:, 2] > common.NEGLIGIBLE_LOG
    colours = tf.where(is_inside_inner_zone, "red", "blue")

    number_in_inner = tf.reduce_sum(tf.cast(is_inside_inner_zone, tf.int64))

    axis_types = analyse.__get_axis_types(cis)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    analyse.__plot_3d_with_axis_types(ax.scatter3D,
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
                                      color=colours.numpy().astype(str))

    print(f"Number in inner zone: {number_in_inner}; in outer: {len(indices)}")
