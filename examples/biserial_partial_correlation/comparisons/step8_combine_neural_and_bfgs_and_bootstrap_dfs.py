import biparcorr_analyse_funcs as biparcorr
from neuralcis import comparisons_funcs as comp
import pandas as pd

df_neural = pd.read_parquet("summaries/summary_neural.parquet")
df_bfgs = pd.read_parquet("summaries/summary_bfgs.parquet")
params = pd.DataFrame(biparcorr.load_params_dict())

df = pd.concat([params.reset_index(drop=True),
                df_neural.reset_index(drop=True),
                df_bfgs.reset_index(drop=True)], axis=1)

comp.save_summary_parquet(f"summary", df)
