import biparcorr_analyse_funcs as biparcorr
from neuralcis import comparisons_funcs as comp


NUM_PARAM_SAMPLES = 500


method_name = "bootstrap_lr"
summary_dict, summary_grids = biparcorr.summarise_pvalues_files(
    method_name=method_name,
    data_type="ps",
    num_params=NUM_PARAM_SAMPLES,
)

summary_name = f"summary_{method_name}"
comp.save_summary_parquet(summary_name, summary_dict)
comp.save_summary_numpy(summary_name, summary_grids)
