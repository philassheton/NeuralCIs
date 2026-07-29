import behfish_analyse_funcs as behfish
from neuralcis import comparisons_funcs as comp

method_name = "welch"
summary_dict, summary_grids = behfish.summarise_pvalues_files(
    method_name=method_name,
    data_type="ps",
)

summary_name = f"summary_{method_name}"
comp.save_summary_parquet(summary_name, summary_dict)
comp.save_summary_numpy(summary_name, summary_grids)
