# Imports used in the script
import pandas as pd
import os
from utils.utils_forecast_data_helper import _load_data_demand
from neuralforecast import NeuralForecast

from utils.utils_artc import (
    filter_rows_sku_fac, _add_weather_data_with_unknown_facility,
    process_long_data, extract_forecasting_res, prepare_weekly_inputs_multivar,
    get_df_context_df_future, 
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

def main():
    overwrite_flag =  True
    initial_processing = True
    horizon =  12
    LAG =  3
    context_size = 12
    freqency = 'M'
    testing_len = 11
    # f"./input_output/resampled_Enrich_filtered_W_aggr_9OCT_sku_facility.csv"

    demand_data_path =  'inputoutput/enriched_output.xlsx'

    df_demand_long = pd.read_excel(demand_data_path) 
    df_demand_long = df_demand_long.rename(columns={"SKU_Country": "unique_id", "Month_Year": "ds", "Sales": "y","Temperature": "temperature_ex", "promo_num": "promo_ex"})
    

    # demand_data_path = f'./inputoutput/sales_nonzero_long.json'
    overwrite_flag =  True
    task = 'CRP12'
    initial_processing =  True

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
        df_demand_long = df_demand_long.rename(columns={"SKU_Country": "unique_id", "Month_Year": "ds", "Sales": "y","Temperature": "temperature_ex", "promo_num": "promo_ex"})
    
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
        DeepAR(h=horizon,
               input_size=context_size,
               lstm_n_layers=3,
               trajectory_samples=100,
               loss=DistributionLoss(distribution='Normal', return_params=False), #DeepAR only supports distributional outputs.
               learning_rate=learning_rate,
               futr_exog_list=futr_exog_list,
               max_steps=training_steps,
               val_check_steps=val_check_steps,
               early_stop_patience_steps=early_stop_patience_steps,
               scaler_type=scaler_type,
               enable_progress_bar=True, 
               random_seed=42),

        DeepNPTS(
            loss=loss,
            h=horizon,
                 input_size=context_size,
                 max_steps=training_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 scaler_type=scaler_type,
                 enable_progress_bar=True, 
                 random_seed=42),
        
        DeepNPTS(loss=loss,
                 h=horizon,
                 input_size=context_size,
                 futr_exog_list=futr_exog_list,
                 max_steps=training_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 scaler_type=scaler_type,
                 enable_progress_bar=True, 
                 random_seed=42),

        DilatedRNN(loss=loss,
                   h=horizon,
                   input_size=context_size,
                #    loss=DistributionLoss(
                #        distribution='Normal', return_params=False),
                   scaler_type=scaler_type,
                   encoder_hidden_size=100,  
                   max_steps=training_steps,
                   val_check_steps=val_check_steps,
                   early_stop_patience_steps=early_stop_patience_steps, 
                random_seed=42),
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

        DLinear(h=horizon,
                input_size=context_size,
                loss=loss,
                scaler_type=scaler_type,
                learning_rate=learning_rate,
                max_steps=training_steps,
                val_check_steps=val_check_steps,
                early_stop_patience_steps=early_stop_patience_steps, 
                random_seed=42),

        RNN(h=horizon,
            input_size=-1,
            inference_input_size=context_size,
            loss=DistributionLoss(distribution='Normal', return_params=False),#better with distributional
            # loss=loss,
            scaler_type=scaler_type,
            encoder_n_layers=2,
            encoder_hidden_size=128,
            context_size=context_size,
            decoder_hidden_size=128,
            decoder_layers=2,
            max_steps=training_steps,
            futr_exog_list=futr_exog_list,
            early_stop_patience_steps=early_stop_patience_steps, 
            random_seed=42
            ),

        TCN(h=horizon,
            input_size=-1,
            loss=DistributionLoss(distribution='Normal', return_params=False), #better with distributional
            # loss=loss,
            learning_rate=learning_rate,
            kernel_size=2,
            dilations=[1, 2, 4, 8, 16],
            encoder_hidden_size=128,
            context_size=context_size,
            decoder_hidden_size=128,
            decoder_layers=2,
            max_steps=training_steps,
            scaler_type=scaler_type,
            futr_exog_list=futr_exog_list,
            early_stop_patience_steps=early_stop_patience_steps, 
            random_seed=42
            )
    
    ]
    # models_all = [
    #         RNN(h=horizon,
    #         input_size=context_size,
    #         inference_input_size=context_size,
    #         loss=DistributionLoss(distribution='Normal', return_params=False),
    #         scaler_type=scaler_type,
    #         encoder_n_layers=2,
    #         encoder_hidden_size=128,
    #         context_size=context_size,
    #         decoder_hidden_size=128,
    #         decoder_layers=2,
    #         max_steps=training_steps,
    #         ),
    #     RNN(h=horizon,
    #         input_size=-1,
    #         inference_input_size=context_size,
    #         loss=DistributionLoss(distribution='Normal', return_params=False),
    #         scaler_type=scaler_type,
    #         encoder_n_layers=2,
    #         encoder_hidden_size=128,
    #         context_size=context_size,
    #         decoder_hidden_size=128,
    #         decoder_layers=2,
    #         max_steps=training_steps,
    #         futr_exog_list=futr_exog_list
    #         ),
    #     LSTM(h=horizon, input_size=-1,
    #          loss=DistributionLoss(distribution='Normal', return_params=False),
    #          scaler_type=scaler_type,
    #          encoder_n_layers=2,
    #          encoder_hidden_size=128,
    #          context_size=context_size,
    #          decoder_hidden_size=128,
    #          decoder_layers=2,
    #          max_steps=training_steps,
    #          ),
    #     LSTM(h=horizon, input_size=-1,
    #          loss=DistributionLoss(distribution='Normal', return_params=False),
    #          scaler_type=scaler_type,
    #          encoder_n_layers=2,
    #          encoder_hidden_size=128,
    #          context_size=context_size,
    #          decoder_hidden_size=128,
    #          decoder_layers=2,
    #          max_steps=training_steps,
    #          futr_exog_list=futr_exog_list
    #          ),
    #     Autoformer(h=horizon,
    #                input_size=context_size,
    #                hidden_size=16,
    #                conv_hidden_size=32,
    #                n_head=2,
    #                loss=loss,
    #                valid_loss=loss,
    #                scaler_type=scaler_type,
    #                learning_rate=learning_rate,
    #                max_steps=training_steps,
    #                val_check_steps=val_check_steps,
    #                early_stop_patience_steps=early_stop_patience_steps),
    #     Autoformer(h=horizon,
    #                input_size=context_size,
    #                hidden_size=16,
    #                conv_hidden_size=32,
    #                n_head=2,
    #                loss=loss,
    #                valid_loss=loss,
    #                futr_exog_list=futr_exog_list,
    #                scaler_type=scaler_type,
    #                learning_rate=learning_rate,
    #                max_steps=training_steps,
    #                val_check_steps=val_check_steps,
    #                early_stop_patience_steps=early_stop_patience_steps),

    #     BiTCN(h=horizon,
    #           input_size=context_size,
    #           loss=loss,
    #           max_steps=training_steps,
    #           scaler_type=scaler_type,
    #           early_stop_patience_steps=early_stop_patience_steps
    #           ),
    #     BiTCN(h=horizon,
    #           input_size=context_size,
    #           loss=loss,
    #           max_steps=training_steps,
    #           futr_exog_list=futr_exog_list,
    #           scaler_type=scaler_type,
    #           early_stop_patience_steps=early_stop_patience_steps
    #           ),

    #     DeepAR(h=horizon,
    #            input_size=context_size,
    #            lstm_n_layers=3,
    #            trajectory_samples=100,
    #            loss=DistributionLoss(
    #                distribution='Normal', return_params=False),
    #            learning_rate=learning_rate,
    #            max_steps=training_steps,
    #            val_check_steps=val_check_steps,
    #            early_stop_patience_steps=early_stop_patience_steps,
    #            scaler_type=scaler_type,
    #            enable_progress_bar=True),
    #     DeepAR(h=horizon,
    #            input_size=context_size,
    #            lstm_n_layers=3,
    #            trajectory_samples=100,
    #            loss=DistributionLoss(
    #                distribution='Normal', return_params=False),
    #            learning_rate=learning_rate,
    #            futr_exog_list=futr_exog_list,
    #            max_steps=training_steps,
    #            val_check_steps=val_check_steps,
    #            early_stop_patience_steps=early_stop_patience_steps,
    #            scaler_type=scaler_type,
    #            enable_progress_bar=True),

    #     DeepNPTS(h=horizon,
    #              input_size=context_size,
    #              max_steps=training_steps,
    #              val_check_steps=val_check_steps,
    #              early_stop_patience_steps=early_stop_patience_steps,
    #              scaler_type=scaler_type,
    #              enable_progress_bar=True),
    #     DeepNPTS(h=horizon,
    #              input_size=context_size,
    #              futr_exog_list=futr_exog_list,
    #              max_steps=training_steps,
    #              val_check_steps=val_check_steps,
    #              early_stop_patience_steps=early_stop_patience_steps,
    #              scaler_type=scaler_type,
    #              enable_progress_bar=True),

    #     DilatedRNN(h=horizon,
    #                input_size=context_size,
    #                loss=DistributionLoss(
    #                    distribution='Normal', return_params=False),
    #                scaler_type=scaler_type,
    #                encoder_hidden_size=100,  # max_steps=training_steps,
    #                val_check_steps=val_check_steps,
    #                early_stop_patience_steps=early_stop_patience_steps),
    #     DilatedRNN(h=horizon,
    #                input_size=context_size,
    #                loss=DistributionLoss(
    #                    distribution='Normal', return_params=False),
    #                scaler_type=scaler_type,
    #                encoder_hidden_size=100,  # max_steps=training_steps,
    #                val_check_steps=val_check_steps,
    #                futr_exog_list=futr_exog_list,
    #                early_stop_patience_steps=early_stop_patience_steps),

    #     DLinear(h=horizon,
    #             input_size=context_size,
    #             loss=loss,
    #             scaler_type=scaler_type,
    #             learning_rate=learning_rate,
    #             max_steps=training_steps,
    #             val_check_steps=val_check_steps,
    #             early_stop_patience_steps=early_stop_patience_steps),

    #     FEDformer(h=horizon,
    #               input_size=context_size,
    #               modes=64,
    #               hidden_size=64,
    #               conv_hidden_size=128,
    #               n_head=8,
    #               loss=loss,
    #               scaler_type=scaler_type,
    #               learning_rate=learning_rate,
    #               max_steps=training_steps,
    #               batch_size=2,
    #               windows_batch_size=32,
    #               val_check_steps=val_check_steps,
    #               early_stop_patience_steps=early_stop_patience_steps),
    #     FEDformer(h=horizon,
    #               input_size=context_size,
    #               modes=64,
    #               hidden_size=64,
    #               conv_hidden_size=128,
    #               n_head=8,
    #               loss=loss,
    #               futr_exog_list=futr_exog_list,
    #               scaler_type=scaler_type,
    #               learning_rate=learning_rate,
    #               max_steps=training_steps,
    #               batch_size=2,
    #               windows_batch_size=32,
    #               val_check_steps=val_check_steps,
    #               early_stop_patience_steps=early_stop_patience_steps),

    #     GRU(h=horizon, input_size=-1,
    #         loss=DistributionLoss(
    #             distribution='Normal', return_params=False),
    #         scaler_type=scaler_type,
    #         encoder_n_layers=2,
    #         encoder_hidden_size=128,
    #         context_size=context_size,
    #         decoder_hidden_size=128,
    #         decoder_layers=2,
    #         max_steps=training_steps,
    #         ),

    #     GRU(h=horizon, input_size=-1,
    #         loss=DistributionLoss(
    #             distribution='Normal', return_params=False),
    #         scaler_type=scaler_type,
    #         encoder_n_layers=2,
    #         encoder_hidden_size=128,
    #         context_size=context_size,
    #         decoder_hidden_size=128,
    #         decoder_layers=2,
    #         max_steps=training_steps,
    #         futr_exog_list=futr_exog_list
    #         ),

    #     # The code you provided is not valid Python code. It seems to be a comment section with the
    #     # text "NHITS" and "
    #     NHITS(h=horizon,
    #           input_size=context_size,
    #           loss=DistributionLoss(distribution='Normal',
    #                                 return_params=False),
    #           max_steps=training_steps,
    #           early_stop_patience_steps=early_stop_patience_steps,
    #           val_check_steps=val_check_steps,
    #           scaler_type=scaler_type,
    #           learning_rate=learning_rate,
    #           valid_loss=sCRPS(level=[80, 90])),

    #     Informer(h=horizon,
    #              input_size=context_size,
    #              loss=loss,
    #              valid_loss=loss,
    #              early_stop_patience_steps=early_stop_patience_steps,
    #              val_check_steps=val_check_steps,
    #              scaler_type=scaler_type,
    #              learning_rate=learning_rate,
    #              max_steps=training_steps,
    #              hidden_size=16,
    #              conv_hidden_size=32,
    #              n_head=2
    #              ),

    #     Informer(h=horizon,
    #              input_size=context_size,
    #              loss=loss,
    #              valid_loss=loss,
    #              early_stop_patience_steps=early_stop_patience_steps,
    #              val_check_steps=val_check_steps,
    #              scaler_type=scaler_type,
    #              learning_rate=learning_rate,
    #              max_steps=training_steps,
    #              hidden_size=16,
    #              conv_hidden_size=32,
    #              n_head=2,
    #              futr_exog_list=futr_exog_list
    #              ),

    #     iTransformer(h=horizon,
    #                  input_size=context_size,
    #                  loss=loss,
    #                  valid_loss=loss,
    #                  early_stop_patience_steps=early_stop_patience_steps,
    #                  val_check_steps=val_check_steps,
    #                  scaler_type=scaler_type,
    #                  learning_rate=learning_rate,
    #                  max_steps=training_steps,
    #                  n_series=len(Y_train_df['unique_id'].unique()),
    #                  hidden_size=128,
    #                  n_heads=2,
    #                  e_layers=2,
    #                  d_layers=1,
    #                  d_ff=4,
    #                  factor=1,
    #                  dropout=0.1,
    #                  use_norm=True,
    #                  batch_size=32),
    #     iTransformer(h=horizon,
    #                  input_size=context_size,
    #                  loss=loss,
    #                  valid_loss=loss,
    #                  early_stop_patience_steps=early_stop_patience_steps,
    #                  val_check_steps=val_check_steps,
    #                  scaler_type=scaler_type,
    #                  learning_rate=learning_rate,
    #                  max_steps=training_steps,
    #                  n_series=len(Y_train_df['unique_id'].unique()),
    #                  hidden_size=128,
    #                  n_heads=2,
    #                  e_layers=2,
    #                  d_layers=1,
    #                  d_ff=4,
    #                  factor=1,
    #                  dropout=0.1,
    #                  use_norm=True,
    #                  batch_size=32,
    #                  futr_exog_list=futr_exog_list),

    #     KAN(h=horizon,
    #         input_size=context_size,
    #         loss=DistributionLoss(
    #             distribution='Normal', return_params=False),
    #         max_steps=training_steps,
    #         scaler_type=scaler_type,
    #         ),
        
    #     KAN(h=horizon,
    #         input_size=context_size,
    #         loss=DistributionLoss(
    #             distribution='Normal', return_params=False),
    #         max_steps=training_steps,
    #         scaler_type=scaler_type,
    #         futr_exog_list=futr_exog_list
    #         ),




    #     MLP(h=horizon, input_size=context_size,
    #         loss=DistributionLoss(
    #             distribution='Normal', return_params=False),
    #         scaler_type=scaler_type,
    #         learning_rate=learning_rate,
    #         max_steps=training_steps,
    #         val_check_steps=val_check_steps,
    #         early_stop_patience_steps=early_stop_patience_steps),
        
    #     NBEATS(h=horizon,
    #            input_size=context_size,
    #            loss=DistributionLoss(
    #                distribution='Normal', return_params=False),
    #            scaler_type=scaler_type,
    #            max_steps=training_steps,
    #            val_check_steps=val_check_steps,
    #            early_stop_patience_steps=early_stop_patience_steps),

    #     NHITS(h=horizon,
    #           input_size=context_size,
    #           loss=DistributionLoss(
    #               distribution='Normal', return_params=False),
    #           n_freq_downsample=[2, 1, 1],
    #           scaler_type=scaler_type,
    #           max_steps=training_steps,
    #           early_stop_patience_steps=early_stop_patience_steps,
    #           inference_windows_batch_size=1,
    #           val_check_steps=val_check_steps,
    #           learning_rate=learning_rate),
        
    #     NLinear(h=horizon,
    #             input_size=context_size,
    #             loss=loss,
    #             scaler_type=scaler_type,
    #             learning_rate=learning_rate,
    #             max_steps=training_steps,
    #             val_check_steps=val_check_steps,
    #             early_stop_patience_steps=early_stop_patience_steps),

    #     PatchTST(h=horizon,
    #              input_size=context_size,
    #              patch_len=24,
    #              stride=24,
    #              revin=False,
    #              hidden_size=16,
    #              n_heads=4,
    #              scaler_type=scaler_type,
    #              loss=DistributionLoss(
    #                  distribution='Normal', return_params=False),
    #              # loss=MAE(),
    #              learning_rate=learning_rate,
    #              max_steps=training_steps,
    #              val_check_steps=val_check_steps,
    #              early_stop_patience_steps=early_stop_patience_steps),

    #     SOFTS(h=horizon,
    #           input_size=context_size,
    #           n_series=len(Y_train_df['unique_id'].unique()),
    #           hidden_size=256,
    #           d_core=256,
    #           e_layers=2,
    #           d_ff=64,
    #           dropout=0.1,
    #           use_norm=True,
    #           loss=loss,
    #           valid_loss=loss,
    #           early_stop_patience_steps=early_stop_patience_steps,
    #           batch_size=32),
    #     SOFTS(h=horizon,
    #           input_size=context_size,
    #           n_series=len(Y_train_df['unique_id'].unique()),
    #           hidden_size=256,
    #           d_core=256,
    #           e_layers=2,
    #           d_ff=64,
    #           dropout=0.1,
    #           use_norm=True,
    #           loss=loss,
    #           valid_loss=loss,
    #           early_stop_patience_steps=early_stop_patience_steps,
    #           batch_size=32,
    #           futr_exog_list=futr_exog_list),

    #     StemGNN(h=horizon,
    #             input_size=context_size,
    #             n_series=len(Y_train_df['unique_id'].unique()),
    #             scaler_type=scaler_type,
    #             max_steps=training_steps,
    #             early_stop_patience_steps=early_stop_patience_steps,
    #             val_check_steps=val_check_steps,
    #             learning_rate=learning_rate,
    #             loss=loss,
    #             valid_loss=loss,
    #             batch_size=32
    #             ),

    #     TCN(h=horizon,
    #         input_size=-1,
    #         loss=DistributionLoss(distribution='Normal', return_params=False),
    #         learning_rate=learning_rate,
    #         kernel_size=2,
    #         dilations=[1, 2, 4, 8, 16],
    #         encoder_hidden_size=128,
    #         context_size=context_size,
    #         decoder_hidden_size=128,
    #         decoder_layers=2,
    #         max_steps=training_steps,
    #         scaler_type=scaler_type,
    #         ),
    #     TCN(h=horizon,
    #         input_size=-1,
    #         loss=DistributionLoss(distribution='Normal', return_params=False),
    #         learning_rate=learning_rate,
    #         kernel_size=2,
    #         dilations=[1, 2, 4, 8, 16],
    #         encoder_hidden_size=128,
    #         context_size=context_size,
    #         decoder_hidden_size=128,
    #         decoder_layers=2,
    #         max_steps=training_steps,
    #         scaler_type=scaler_type,
    #         futr_exog_list=futr_exog_list
    #         ),

    #     TFT(h=horizon, input_size=context_size,
    #         hidden_size=20,
    #         loss=DistributionLoss(distribution='Normal', return_params=False),
    #         learning_rate=learning_rate,
    #         max_steps=training_steps,
    #         val_check_steps=val_check_steps,
    #         early_stop_patience_steps=early_stop_patience_steps,
    #         scaler_type=scaler_type,
    #         windows_batch_size=None,
    #         enable_progress_bar=True),
    #     TFT(h=horizon, input_size=context_size,
    #         hidden_size=20,
    #         loss=DistributionLoss(distribution='Normal', return_params=False),
    #         learning_rate=learning_rate,
    #         futr_exog_list=futr_exog_list,
    #         max_steps=training_steps,
    #         val_check_steps=val_check_steps,
    #         early_stop_patience_steps=early_stop_patience_steps,
    #         scaler_type=scaler_type,
    #         windows_batch_size=None,
    #         enable_progress_bar=True),

    #     TiDE(h=horizon,
    #          input_size=context_size,
    #          loss=loss,
    #          max_steps=training_steps,
    #          scaler_type=scaler_type,
    #          valid_loss=loss,
    #          val_check_steps=val_check_steps,
    #          early_stop_patience_steps=early_stop_patience_steps,
    #          ),
    #     TiDE(h=horizon,
    #          input_size=context_size,
    #          loss=loss,
    #          max_steps=training_steps,
    #          scaler_type=scaler_type,
    #          valid_loss=loss,
    #          val_check_steps=val_check_steps,
    #          early_stop_patience_steps=early_stop_patience_steps,
    #          futr_exog_list=futr_exog_list
    #          ),

    #     TimesNet(h=horizon,
    #              input_size=context_size,
    #              hidden_size=16,
    #              conv_hidden_size=32,
    #              loss=loss,
    #              scaler_type=scaler_type,
    #              learning_rate=learning_rate,
    #              max_steps=training_steps,
    #              val_check_steps=val_check_steps,
    #              early_stop_patience_steps=early_stop_patience_steps),
    #     TimesNet(h=horizon,
    #              input_size=context_size,
    #              hidden_size=16,
    #              conv_hidden_size=32,
    #              loss=loss,
    #              futr_exog_list=futr_exog_list,
    #              scaler_type=scaler_type,
    #              learning_rate=learning_rate,
    #              max_steps=training_steps,
    #              val_check_steps=val_check_steps,
    #              early_stop_patience_steps=early_stop_patience_steps),

    #     TSMixer(h=horizon,
    #             input_size=context_size,
    #             n_series=len(Y_train_df['unique_id'].unique()),
    #             n_block=4,
    #             ff_dim=4,
    #             dropout=0,
    #             revin=True,
    #             scaler_type=scaler_type,
    #             max_steps=training_steps,
    #             early_stop_patience_steps=early_stop_patience_steps,
    #             val_check_steps=val_check_steps,
    #             learning_rate=learning_rate,
    #             loss=loss,
    #             valid_loss=loss,
    #             batch_size=32
    #             ),
    #     TSMixerx(h=horizon,
    #              futr_exog_list=futr_exog_list,
    #              input_size=context_size,
    #              n_series=len(Y_train_df['unique_id'].unique()),
    #              n_block=4,
    #              ff_dim=4,
    #              dropout=0,
    #              revin=True,
    #              scaler_type=scaler_type,
    #              max_steps=training_steps,
    #              early_stop_patience_steps=early_stop_patience_steps,
    #              val_check_steps=val_check_steps,
    #              learning_rate=learning_rate,
    #              loss=loss,
    #              valid_loss=loss,
    #              batch_size=32
    #              ),

    #     TimeMixer(h=horizon,
    #               input_size=context_size,
    #               n_series=len(Y_train_df['unique_id'].unique()),
    #               scaler_type=scaler_type,
    #               max_steps=training_steps,
    #               early_stop_patience_steps=early_stop_patience_steps,
    #               val_check_steps=val_check_steps,
    #               learning_rate=learning_rate,
    #               loss=loss,
    #               valid_loss=loss,
    #               batch_size=32
    #               ),

    #     VanillaTransformer(h=horizon,
    #                        input_size=context_size,
    #                        hidden_size=16,
    #                        conv_hidden_size=32,
    #                        n_head=2,
    #                        loss=loss,
    #                        valid_loss=loss,
    #                        scaler_type=scaler_type,
    #                        learning_rate=learning_rate,
    #                        max_steps=training_steps,
    #                        val_check_steps=val_check_steps,
    #                        early_stop_patience_steps=early_stop_patience_steps),
    #     VanillaTransformer(h=horizon,
    #                        input_size=context_size,
    #                        hidden_size=16,
    #                        conv_hidden_size=32,
    #                        n_head=2,
    #                        loss=loss,
    #                        valid_loss=loss,
    #                        futr_exog_list=futr_exog_list,
    #                        scaler_type=scaler_type,
    #                        learning_rate=learning_rate,
    #                        max_steps=training_steps,
    #                        val_check_steps=val_check_steps,
    #                        early_stop_patience_steps=early_stop_patience_steps),
    # ]

    # training models if overwrite
    nf = NeuralForecast(
        models=models_all,
        freq=freqency)
    if overwrite_flag:
        nf.fit(df=Y_train_df, val_size=horizon)
        nf.save(path=nf_path,
                model_index=None,
                overwrite=True,
                save_dataset=True)
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
    y_all_ml_uni, y_all_ml_ex, y_all_nf, y_all_stats = [], [], [], []
    for rolling_id in range(testing_len - LAG + 1):
        df_context, df_future = get_df_context_df_future(
            df, rolling_id, horizon, LAG, context_size, testing_len=testing_len)
        df_context['ds'] = pd.to_datetime(df_context['ds'])
        df_future['ds'] = pd.to_datetime(df_future['ds'])
        
        # NeuralForecast predictions
        y_nf = nf.predict(df=df_context, futr_df=df_future)
        y_all_nf.append(y_nf)
        
    # Combine predictions
    model_names = get_nf_model_names(nf)
    df_pred_all_nf = extract_forecasting_res(y_all_nf, model_names, horizon)

    df_all = pd.concat([df, df_pred_all_nf], axis=1)

    # save s2 results
    save_dataframe_to_json(df=df_all, file_path=file_path_s2)
    df_all.to_csv(file_path_s2.replace('.json', '.csv'))


if __name__ == "__main__":
    main()
