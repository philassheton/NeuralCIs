import biparcorr_analyse_funcs as biparcorr
from neuralcis import comparisons_funcs as comp
import pandas as pd

df_neural = pd.read_parquet("summaries/summary_neural.parquet")
df_neural_500 = pd.read_parquet("summaries/summary_neural_500.parquet")
df_bfgs = pd.read_parquet("summaries/summary_bfgs.parquet")
df_bootstrap_lr = pd.read_parquet("summaries/summary_bootstrap_lr.parquet")
params = pd.DataFrame(biparcorr.load_params_dict())

# The dataframe already restricts us to 500 p-values per parameter but still
# covers the whole 1000 parameter range.  So trim that to just the first 500
# parameters
df_neural_500 = df_neural_500.iloc[0:500, :]
df_bootstrap_lr = df_bootstrap_lr.iloc[0:500, :]
params_500 = params.iloc[0:500, :]

df_chisq = pd.concat([params.reset_index(drop=True),
                      df_neural.reset_index(drop=True),
                      df_bfgs.reset_index(drop=True)], axis=1)

df_bootstrap_lr = pd.concat([params_500.reset_index(drop=True),
                             df_neural_500.reset_index(drop=True),
                             df_bootstrap_lr.reset_index(drop=True)], axis=1)

comp.save_summary_parquet(f"summary_neural_vs_chisq", df_chisq)
comp.save_summary_parquet(f"summary_neural_vs_bootstrap_lr", df_bootstrap_lr)
