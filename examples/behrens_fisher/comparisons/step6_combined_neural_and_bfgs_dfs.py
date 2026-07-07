import behfish_analyse_funcs as behfish
import pandas as pd

df_neural = pd.read_parquet("summaries/summary_neural.parquet",
                            engine="pyarrow")
df_welch = pd.read_parquet("summaries/summary_welch.parquet",
                          engine="pyarrow")
params = pd.DataFrame(behfish.load_params_dict())

df = pd.concat([params.reset_index(drop=True),
                df_neural.reset_index(drop=True),
                df_welch.reset_index(drop=True)], axis=1)

df.to_parquet(f"summaries/summary.parquet",
              engine="pyarrow",
              compression="zstd",
              index=False)
