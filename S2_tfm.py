import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'
import timesfm
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import struct
from datetime import date, datetime
import datetime as dt
import pickle
# from HPL_utils import *
from sklearn.cluster import KMeans
#%%
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from jax._src import config
import random
from IPython.display import clear_output
random.seed(0)
np.random.seed(0)
from utils.utils_artc import string_to_number_list,  fill_none_with_ffill_bfill, save_dataframe_to_json
import sys
use_scaler, timesfm_backend = True, 'gpu'
config.update("jax_platforms", {"cpu": "cpu", "gpu": "cuda", "tpu": ""}[timesfm_backend])


def main():
    model_id = 1
    # horizon =  12
    # LAG =  3
    # context_size = 12
    # freqency = 'M'
    # testing_len = 11
    # task = 'CRP12'
    # identifier = f'/mnt/c/data/CRP12/KC_monthly/{task}_context_size_{context_size}_horizon_{horizon}_testing_len_{testing_len}_KC_Lag3_deep_robust_all'
 

    horizon = int(sys.argv[1])  # 13
    context_size = int(sys.argv[2])  # 128
    testing_len = int(sys.argv[3])  # 13
    identifier = sys.argv[4]

    file_path_s2 = f'{identifier}/df_all_mixing_training_s2.json'
    file_path_s3 = f'{identifier}/df_all_mixing_training_s3.json'
    file_path_s3_temp = f'{identifier}/df_all_mixing_training_s3_temp.json'
    freq_val = 1 
    
    # load s2 data, where nf forecast done already
    df_loaded_all = pd.read_json(file_path_s2, orient='records', lines=True)
    if model_id == 1:
        print(f"=========================model_id = {model_id}, making the one time TIMESFM forecasting=========================")
        #load model
        # model = timesfm.TimesFm(
        #     context_len=context_size,
        #     horizon_len=horizon,
        #     input_patch_len=32,
        #     output_patch_len=128,
        #     num_layers=20,
        #     model_dims=1280,
        #     backend=timesfm_backend
        # )
        # model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
            # For Torch
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=12,
        ),
        
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
            )
        
        model = tfm
        
        
        
        inputs = df_loaded_all['context'].to_list()
        inputs = [string_to_number_list(e) if isinstance(e, str) else e for e in inputs]
        # inputs = [fill_none_with_ffill_bfill(input) for input in inputs]
        
        #update df_loaded_all with None removed by ffill and bfill
        df_loaded_all['context'] = inputs
        raw_forecast, _ = model.forecast(
            inputs=inputs, freq=[freq_val] * len(inputs)
        )
        # save resutls by mergin gwith previous one
        df_loaded_all['timefm'] = raw_forecast.tolist()
        # df_loaded_all.to_json(file_path_s3, orient='records', lines=True, date_format='iso', double_precision=15)

    else:
        print(f"=========================model_id = {model_id} , loaded from temp=========================")
        df_s3 = pd.read_json(file_path_s3_temp, orient='records', lines=True)
        df_loaded_all['timefm'] = df_s3['timefm']
        
    save_dataframe_to_json(df=df_loaded_all, file_path=file_path_s3)
    df_loaded_all.to_csv(file_path_s3.replace('.json', '.csv'))
if __name__ == "__main__":
    main()




