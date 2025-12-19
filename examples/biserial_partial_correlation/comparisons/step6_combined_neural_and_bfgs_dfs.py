import biparcorr_analyse_funcs as biparcorr
import pandas as pd

df_neural = pd.read_parquet("summaries/summary_neural.parquet",
                            engine="pyarrow")
df_bfgs = pd.read_parquet("summaries/summary_bfgs.parquet",
                          engine="pyarrow")
params = pd.DataFrame(biparcorr.load_params_dict())

df = pd.concat([params.reset_index(drop=True),
                df_neural.reset_index(drop=True),
                df_bfgs.reset_index(drop=True)], axis=1)

df.to_parquet(f"summaries/summary.parquet",
              engine="pyarrow",
              compression="zstd",
              index=False)
