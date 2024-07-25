import tensorflow as tf
import matplotlib.pyplot as plt

from neuralcis import NeuralCIs


def visualize_sampling_fn(
        cis: NeuralCIs,
        num_points: int = 100,
        **param_values,
):

    """Visualize the sampling and contrast functions at a given set of param
    values.  Estimate variables are plotted pair by pair.  If 
    a given variable is not provided in param_values, the median of the 
    distribution for that param will be used.

    Example:

        visualize_sampling_and_contrast(cis, mu=0., n=4.)
    """

    for name, dist in zip(cis.param_names_in_sim_order,
                          cis.param_dists_in_sim_order):
        if name not in param_values:
            param_values[name] = dist.from_std_uniform(0.5).numpy()

    params = {}
    for name in cis.param_names_in_sim_order:
        params[name] = tf.repeat(param_values[name], num_points)
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

    title = "Distribution of estimates with params set at:"
    spacer = " "
    for name, value in param_values.items():
        title += f"{spacer}{name}={value:.2f}"
        spacer = "; "
    title += "."
    fig.suptitle(title, wrap=True)

    fig.show()
