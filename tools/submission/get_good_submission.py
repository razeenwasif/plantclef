import pandas as pd
import json

try:
    df = pd.read_csv('src_experiments/011_bioclip25_ssl_adaptive/outputs/ssl_last_blocks_adaptive_infer/softmax_mean_gap0.6/submission.csv')
    print(df.head(2).to_json(orient='records'))
except Exception as e:
    print(e)
