# Imports used in the script
import pandas as pd
import os
from utils.utils_forecast_data_helper import _load_data_demand
from neuralforecast import NeuralForecast

from utils.utils_artc import (
    filter_rows_sku_fac, _add_weather_data_with_unknown_facility,
    process_long_data, extract_forecasting_res, prepare_weekly_inputs_multivar,
    get_df_context_df_future,get_df_context_df_future_shap,
    get_nf_model_names, save_dataframe_to_json, prepare_training_data_multivar, extend_weather_data_for_facilities
)


import random
import numpy as np
random.seed(0)
np.random.seed(0)

from neuralforecast.losses.pytorch import (
    MAE,
    MSE,
    DistributionLoss,
    sCRPS,
)

from neuralforecast.models import (
    TimeMixer,
    TSMixer,
    Autoformer,
    BiTCN,
    DeepAR,
    DeepNPTS,
    DilatedRNN,
    DLinear,
    FEDformer,
    GRU,
    NHITS,
    Informer,
    iTransformer,
    KAN,
    LSTM,
    MLP,
    NBEATS,
    NLinear,
    PatchTST,
    RNN,
    SOFTS,
    StemGNN,
    TCN,
    TFT,
    TiDE,
    TimesNet,
    VanillaTransformer,
    TSMixerx,
)

import sys
# 4. Function to build modified future_df with masked features
def modify_future(future_df, active_features,ref_temperature,ref_promo):
    df_mod = future_df.copy()
    if 'temperature_ex' not in active_features:
        df_mod['temperature_ex'] = ref_temperature
    if 'promo_ex' not in active_features:
        df_mod['promo_ex'] = ref_promo
    return df_mod

def main():
    overwrite_flag =  True
    horizon =  12
    LAG =  3
    context_size = 12
    freqency = 'M'
    testing_len = 11
    # f"./input_output/resampled_Enrich_filtered_W_aggr_9OCT_sku_facility.csv"

    demand_data_path =  'inputoutput/enriched_output.xlsx'
    file_path_group_shap = 'inputoutput/DilatedRNN_ex/group_shap.json'

    df_demand_long = pd.read_excel(demand_data_path) 
    df_demand_long = df_demand_long.rename(columns={"SKU_Country": "unique_id", "Month_Year": "ds", "Sales": "y","Temperature": "temperature_ex", "promo_num": "promo_ex"})
    
    # demand_data_path = f'./inputoutput/sales_nonzero_long.json'
    overwrite_flag =  True
    task = 'CRP12'
    initial_processing =  False

    identifier = f'/mnt/c/data/CRP12/KC_monthly/{task}_context_size_{context_size}_horizon_{horizon}_testing_len_{testing_len}_KC_Lag3_deep_robust_all'

    ##============================================================================== end parameters ========================
    file_path_Y_train_df = f'{identifier}/Y_train_df.json'
    file_path_s1 = f'{identifier}/df_all_mixing_training_s1.json'
    file_path_s2 = f'{identifier}/df_all_mixing_training_s2.json'
    nf_path = f'{identifier}/nf'
    
    if initial_processing:
        # load demand data and temperature data
        # df_temp_sales = pd.read_csv(demand_data_path) # 41965 rows x 6 column
        df_demand_long = pd.read_excel(demand_data_path) 
        df_demand_long = df_demand_long.rename(columns={"SKU_Country": "unique_id", "Month_Year": "ds", "Sales": "y","Temperature": "temperature_ex", "promo": "promo_ex"})
    
        df_demand_long['ds'] = pd.to_datetime(df_demand_long['ds'])
        # date_ealiest = df_demand_long.ds.min()

        # Monthly aggregated data processing
        df_sales = process_long_data(
            df_demand_long, freq= freqency, index='ds', sku_column='unique_id', target_column='y').T  # [103 rows x 235 columns]
        df_price = process_long_data(
            df_demand_long, freq= freqency, index='ds', sku_column='unique_id', target_column='promo_ex') #[235 rows x 103 columns]
        df_temperature = process_long_data(
            df_demand_long, freq= freqency, index='ds', sku_column='unique_id', target_column='temperature_ex') #[235 rows x 103 columns]

        # Fill missing values
        df_sales = df_sales.bfill(axis=1).ffill(axis=1)
        df_price = df_price.bfill(axis=1).ffill(axis=1)
        df_temperature = df_temperature.bfill(axis=1).ffill(axis=1)

        # Assertions to ensure no missing values remain
        assert not df_sales.isnull().values.any(), "check df_sales"
        assert not df_price.isnull().values.any(), "check df_price"
        assert not df_temperature.isnull().values.any(), "check df_temperature"

        # set the start_ori based on rolling numbers
        # '2021-09-26'
        testing_context_start_date_ori = df_sales.columns[-testing_len-context_size]
        training_start_date, training_end_date = df_sales.columns[0], df_sales.columns[-testing_len-1]
        training_date_range = pd.date_range(
            start=training_start_date, end=training_end_date, freq=freqency)

        if overwrite_flag:
            idx = df_sales.columns.to_list().index(training_end_date)
            df_train_sales = df_sales.iloc[:, :idx+1]
            Y_train_df = prepare_training_data_multivar(
                df_train_sales, df_temperature,  df_price, training_date_range)
            save_dataframe_to_json(df=Y_train_df, file_path=file_path_Y_train_df)
            Y_train_df.to_csv(file_path_Y_train_df.replace('.json', '.csv'))

        if os.path.exists(file_path_Y_train_df):
            Y_train_df = pd.read_json(
                file_path_Y_train_df, orient='records', lines=True)
            Y_train_df['ds'] = pd.to_datetime(Y_train_df['ds'], errors='coerce')
        else:
            idx = df_sales.columns.to_list().index(training_end_date)
            df_train_sales = df_sales.iloc[:, :idx+1]
            Y_train_df = prepare_training_data_multivar(
                df_train_sales, df_temperature,  df_price, training_date_range)
            save_dataframe_to_json(df=Y_train_df, file_path=file_path_Y_train_df)
            Y_train_df.to_csv(file_path_Y_train_df.replace('.json', '.csv'))

        # prepare the testing set, specified by testing_len
        testing_monthly = prepare_weekly_inputs_multivar(
            df_sales=df_sales,
            df_temperature=df_temperature,
            df_price=df_price,
            testing_context_start_date_ori=testing_context_start_date_ori,
            horizon=horizon,
            LAG=LAG,
            context_size=context_size,
            testing_len=testing_len)
        # save the file
        save_dataframe_to_json(df=testing_monthly, file_path=file_path_s1)
        testing_monthly.to_csv(file_path_s1.replace('.json', '.csv'))
    else:
        print('===============', file_path_Y_train_df)
        Y_train_df = pd.read_json(file_path_Y_train_df, orient='records', lines=True)
        Y_train_df['ds'] = pd.to_datetime(Y_train_df['ds'], errors='coerce')
        testing_monthly = pd.read_json(file_path_s1, orient='records', lines=True)
        
    #training models
    training_steps = 1000
    val_check_steps = 500
    early_stop_patience_steps = 300
    scaler_type = 'robust'
    loss = MAE()
    learning_rate = 1e-3
    futr_exog_list = ['temperature_ex', 'promo_ex']
    models_all = [
        DilatedRNN(loss=loss,
                   h=horizon,
                   input_size=context_size,
                #    loss=DistributionLoss(
                #        distribution='Normal', return_params=False),
                   scaler_type=scaler_type,
                   encoder_hidden_size=100, 
                   max_steps=training_steps,
                   val_check_steps=val_check_steps,
                   futr_exog_list=futr_exog_list,
                   early_stop_patience_steps=early_stop_patience_steps, 
                   random_seed=42),
    ]

    # nf = NeuralForecast(
    #     models=models_all,
    #     freq=freqency)
    # if overwrite_flag:
    #     nf.fit(df=Y_train_df, val_size=horizon)
    #     nf.save(path=nf_path,
    #             model_index=None,
    #             overwrite=True,
    #             save_dataset=True)
    # load models if trained
    if not os.path.exists(nf_path):
        os.makedirs(nf_path)
    # training or loading
    try:
        nf = NeuralForecast.load(path=nf_path)
    except:
        print("model  not trained, retraining")
        # train models
        nf.fit(df=Y_train_df, val_size=horizon)
        nf.save(path=nf_path,
                model_index=None,
                overwrite=True,
                save_dataset=True)

    # Generate predictions
    df = testing_monthly.copy()
    df_testing_last = (df.groupby('unique_id', sort=False)
             .tail(1)
             .reset_index(drop=True))


    # Create an empty DataFrame with specified columns
    shap_results_group = pd.DataFrame(columns=['unique_id', 'ds','y', 'temperature_ex', 'promo_ex','DilatedRNN','DilatedRNN_Base','temperature_ex_shap','promo_ex_shap'])
    accuracy_results_group = []
    for unique_id in df_testing_last['unique_id'].unique().tolist():
        # 1. Filter for the specific unique_id if needed
        df_target = df_testing_last[df_testing_last['unique_id'] == unique_id].copy()
        # for rolling_id in range(testing_len - LAG + 1):
        df_context, df_future = get_df_context_df_future_shap(
            df_target, horizon, context_size, testing_len=testing_len)
        df_context['ds'] = pd.to_datetime(df_context['ds'])
        df_future['ds'] = pd.to_datetime(df_future['ds'])

        # context_df = df_target_test.iloc[i: context_size + i]  # last 12 rows before the last 3
        # future_df = df_target_test.iloc[context_size + i : context_size + i + horizon] 
        
        shap_results_eachdate = df_future.copy()
        future_df_grouped = df_future.groupby('unique_id').agg({
            'ds': list,
            'temperature_ex': list,
            'promo_ex': list,
        }).reset_index()
        # future_df_grouped['target_date'] = future_df_grouped['ds'][0][3]
        # future_df_grouped['y_lag2'] = future_df_grouped['y'][0][3]
        
        # future_df = df_future.reset_index(drop=True).drop(columns=["y"], errors='ignore')
        
        ## get the prediction on model
        pred = nf.predict(df=df_context, futr_df=df_future)
        shap_results_eachdate['DilatedRNN'] = pred['DilatedRNN'].astype(float).round(2).values
        future_df_grouped['DilatedRNN'] = [pred['DilatedRNN'].astype(float).round(2).values.tolist()]
        # future_df_grouped['DilatedRNN_pred_lag3'] = future_df_grouped['DilatedRNN'][0][3]

        ## get the expected prediction on model
        # 3. Define reference values
        ref_temperature = df_future['temperature_ex'].mean()
        ref_promo = df_future['promo_ex'].mean()
        
        df_exp = df_future.copy()
        df_exp['temperature_ex'] = ref_temperature
        df_exp['promo_ex'] = ref_promo

        pred = nf.predict(df=df_context, futr_df=df_exp)
        shap_results_eachdate['DilatedRNN_Base'] = pred['DilatedRNN'].astype(float).round(2).values
        future_df_grouped['DilatedRNN_Base'] = [pred['DilatedRNN'].astype(float).round(2).values.tolist()]

        # 5. Compute SHAP values
        # shap_results = []
        for feature in ['temperature_ex', 'promo_ex']:
            phi = []

            for subset in [[], ['temperature_ex'], ['promo_ex']]:
                if feature in subset:
                    continue

                # S = subset, S ∪ {i} = with feature
                subset_with_i = subset + [feature]

                fut_S = modify_future(df_future, subset,ref_temperature,ref_promo)
                fut_S_i = modify_future(df_future, subset_with_i,ref_temperature,ref_promo)

                # nf.forecast(X_df=fut_S).reset_index()
                pred_S = nf.predict(df=df_context, futr_df=fut_S)
                pred_S_i = nf.predict(df=df_context, futr_df=fut_S_i)

                # print("subset", subset)
                # print("pred_S", pred_S)
                # print('subset_with_i', subset_with_i)
                # print('pred_S_i', pred_S_i)
                # Weighted difference (1/2 since only 2 features)
                marginal_contrib = (pred_S_i['DilatedRNN'].astype(float).round(2) - pred_S['DilatedRNN'].astype(float).round(2))
                weight = 0.5
                phi.append(marginal_contrib * weight)
                
            
            future_df_grouped[feature + '_shap'] = [sum(phi).astype(float).round(2).values.tolist()]
            shap_results_eachdate[feature + '_shap'] = sum(phi).astype(float).round(2).values.tolist()
            # print('shap_results',shap_results_eachdate)
            # shap_results.append({
            #     'feature': feature,
            #     feature +'shap_value': sum(phi)
            # })
        shap_results_group = pd.concat([shap_results_group,future_df_grouped], axis=0)
        
        # feature_names = ['temperature_ex', 'promo_ex']
        # shap_value_names = ['temperature_ex_shap', 'promo_ex_shap']

        save_dataframe_to_json(df=shap_results_group, file_path=file_path_group_shap )
        shap_results_group.to_csv(file_path_group_shap .replace('.json', '.csv'))



if __name__ == "__main__":
    main()

#Todo: only load the saved model and give the specific SKU name and the month

#Todo: task2 get the result on this setting input.
