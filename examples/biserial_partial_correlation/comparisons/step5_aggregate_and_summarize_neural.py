import biparcorr_analyse_funcs as biparcorr
from neuralcis import comparisons_funcs as comp

method_name = "neural"
summary_dict, summary_grids = biparcorr.summarise_pvalues_files(
    method_name=method_name,
    data_type="ps",
)

summary_name = f"summary_{method_name}"
comp.save_summary_parquet(summary_name, summary_dict)
comp.save_summary_numpy(summary_name, summary_grids)
