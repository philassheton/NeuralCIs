import behfish_analyse_funcs as behfish

import os
import pandas as pd
import numpy as np


method_name = "neural"
summary_dict, summary_grids = behfish.summarise_pvalues_files(
    method_name=method_name,
    data_type="ps",
)

os.makedirs("summaries", exist_ok=True)

summary_df = pd.DataFrame(summary_dict)
summary_df.to_parquet(f"summaries/summary_{method_name}.parquet",
                      engine="pyarrow",
                      compression="zstd",
                      index=False)
np.save(f"summaries/summary_{method_name}.npy", summary_grids)
