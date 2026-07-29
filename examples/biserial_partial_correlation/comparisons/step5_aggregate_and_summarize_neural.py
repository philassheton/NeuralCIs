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


# Repeat but for only first 500 p-values, for comparison with bootstrap LR
summary_dict_500, summary_grids_500 = biparcorr.summarise_pvalues_files(
    method_name=method_name,
    data_type="ps",
    only_first_n_pvalues=500,
)

summary_name_500 = f"summary_{method_name}_500"
comp.save_summary_parquet(summary_name_500, summary_dict_500)
comp.save_summary_numpy(summary_name_500, summary_grids_500)
