import biparcorr_analyse_funcs as biparcorr
from neuralcis import comparisons_funcs as comp


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

params = biparcorr.load_params_dict()
params = {k:p[0:NUM_PARAM_SAMPLES].numpy() for k, p in params.items()}

summary_dict = summary_dict_bootlr | summary_dict_neural | params
comp.save_summary_parquet(f"summary_{method_name}", summary_dict)
