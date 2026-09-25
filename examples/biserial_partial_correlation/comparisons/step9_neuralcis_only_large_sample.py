import biparcorr_analyse_funcs as biparcorr
from examples import neural_alone_run_funcs as run
from examples import comparisons_funcs as comp
from examples import latent_correlation_funcs as lcor

import numpy as np
import pandas as pd

# typing
from typing import Dict
from tensor_annotations.tensorflow import Tensor1, Tensor3
from tensor_annotations.tensorflow import float32 as tf32
from examples import Batch, Samples, Stats


def stats_human_from_samples_raw(
        params_human: Dict[str, Tensor1[tf32, Batch]],
        samples_raw: Tensor3[tf32, Batch, Samples, Stats]
) -> Dict[str, Tensor1[tf32, Batch]]:

    rs = lcor.estimate_correlations_safe_3d(samples_raw, params_human["n"])
    prop_a_hat = lcor.estimate_prop_a(samples_raw, params_human["n"])
    stats_human = {"rho_ab_hat": rs[:, 0],
                   "rho_bc_hat": rs[:, 1],
                   "rho_ac_hat": rs[:, 2],
                   "prop_a_hat": prop_a_hat}
    return stats_human


if __name__ == "__main__":
    params = lcor.sample_params_neural_alone(
        ["rho_ab_partial", "rho_bc", "rho_ac", "prop_a"],
        param_is_prop=[False, False, False, True],
    )
    summary_dicts = run.run_neural_ps(params,
                                      biparcorr.sampling_distribution_fn_raw,
                                      stats_human_from_samples_raw,
                                      len(params["n"]))
    summary_dict = {k:np.array([d[k] for d in summary_dicts])
                    for k in summary_dicts[0].keys()}
    summary_df = pd.DataFrame(summary_dict)
    params_df = pd.DataFrame(params)
    summary_df = pd.concat([summary_df, params_df], axis=1)
    comp.save_summary_parquet("summary_neural_large_sample", summary_df)
