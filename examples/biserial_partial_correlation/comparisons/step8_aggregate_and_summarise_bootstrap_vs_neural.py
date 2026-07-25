import biparcorr_analyse_funcs as biparcorr

import pandas as pd


NUM_PARAM_SAMPLES = 500


method_name = "bootstrap_lr"
summary_dict_bootlr, summary_grids_bootlr = biparcorr.summarise_pvalues_files(
    method_name=method_name,
    data_type="ps",
    num_params=NUM_PARAM_SAMPLES,
)

summary_dict_neural, summary_grids_neural = biparcorr.summarise_pvalues_files(
    method_name="neural",
    data_type="ps",
    num_params=NUM_PARAM_SAMPLES,
    only_first_n_pvalues=500,
)

summary_dict = summary_dict_bootlr | summary_dict_neural
summary_df = pd.DataFrame(summary_dict)

summary_df.to_parquet(f"summaries/summary_{method_name}.parquet",
                      engine="pyarrow",
                      compression="zstd",
                      index=False)
