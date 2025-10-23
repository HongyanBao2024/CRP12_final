import pandas as pd
import numpy as np
import itertools
from neuralforecast.models import RNN, DeepAR, DeepNPTS,DilatedRNN
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import (
    MAE,
    MSE,
    DistributionLoss,
    sCRPS,
)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
from utils.utils_artc import save_dataframe_to_json

model_name = ['DeepNPTS_ex','DilatedRNN_ex','uni2ts_ex']

# file_path_group_shap = 'inputoutput/DeepNPTS_ex/group_shap.json'
# file_path_group_accuracy = 'inputoutput/DeepNPTS_ex/group_accuracy.json'
file_path_group_shap = 'inputoutput/DilatedRNN_ex/group_shap.json'
file_path_group_accuracy = 'inputoutput/DilatedRNN_ex/group_accuracy.json'

import pandas as pd
import numpy as np

# 4. Function to build modified future_df with masked features
def modify_future(future_df, active_features):
    df_mod = future_df.copy()
    if 'temperature_ex' not in active_features:
        df_mod['temperature_ex'] = ref_temperature
    if 'promo_ex' not in active_features:
        df_mod['promo_ex'] = ref_promo
    return df_mod

def Accuracy_adjusted_MAPE_TCCC(y_true, y_pred):
    """
    for TCCC use
    """

    # Replace negative values with 0
    # y_pred[y_pred < 0] = 0

    # Calculate the numerator: sum of absolute differences between actual and predicted values
    numerator = sum(abs(y_j - y_hat_j) for y_j, y_hat_j in zip(y_true, y_pred))

    # Calculate the denominator: sum of the actual and predicted values, scaled by 1/2
    denominator = sum(abs(y_j) for y_j in y_true)

    # Calculate the WFA_market using the max condition
    Accuracy_adjusted_MAPE = max(1 - (numerator / denominator), 0)

    return Accuracy_adjusted_MAPE


def calculate_rowwise_accuracy(y_true, y_pred):
    # Convert inputs to NumPy arrays
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Initialize accuracy array
    acc = np.zeros_like(y_true, dtype=float)

    # Handle cases where y_true == 0 and y_pred == 0
    mask_zero_both = (y_true == 0) & (y_pred == 0)
    acc[mask_zero_both] = 1

    # Handle cases where y_true == 0 and y_pred != 0
    mask_zero_true = (y_true == 0) & (y_pred != 0)
    acc[mask_zero_true] = 0

    # Handle normal cases where y_true != 0
    mask_non_zero = y_true != 0
    acc[mask_non_zero] = 1 - abs(y_pred[mask_non_zero] -
                                y_true[mask_non_zero]) / abs(y_true[mask_non_zero])
    acc = np.maximum(acc, 0)

    # Convert to list and return
    return acc.tolist()

# demand_data_path =  'inputoutput/enriched_output.xlsx'
demand_data_path =  'inputoutput/enriched_output_copy.xlsx'

df_demand_long = pd.read_excel(demand_data_path) 
df_demand_long = df_demand_long.rename(columns={"SKU_Country": "unique_id", "Month_Year": "ds", "Sales": "y","Temperature": "temperature_ex", "promo_num": "promo_ex"})

# Convert ds to datetime
df_demand_long['ds'] = pd.to_datetime(df_demand_long['ds'], dayfirst=True)
df = pd.read_csv('inputoutput/enriched_output.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)
adjusting_ratio = 0.8
futr_exog_cols = ['temperature_ex', 'promo_ex']
horizon = 12
context_size = 12
num_train = 37
num_test = 9
# models = [
#     RNN(h=horizon, input_size=context_size, loss=MAE(), futr_exog_list=futr_exog_cols, max_steps=500),
#     # TCN(h=horizon, input_size=context_size, loss=MAE(), futr_exog_list=futr_exog_cols, max_steps=1),
#     # DeepNPTS(h=horizon, input_size=context_size, loss=MAE(), futr_exog_list=futr_exog_cols, max_steps=1),
#     # DeepAR(h=horizon, input_size=context_size, loss=MAE(), futr_exog_list=futr_exog_cols, max_steps=1)
# ]
training_steps = 1000
val_check_steps = 500
early_stop_patience_steps = 300
scaler_type = 'robust'
loss = MAE()
learning_rate = 1e-3
futr_exog_list = ['temperature_ex', 'promo_ex']
models = [
    # DeepAR(h=horizon,
    #            input_size=context_size,
    #            lstm_n_layers=3,
    #            trajectory_samples=100,
    #            loss=DistributionLoss(distribution='Normal', return_params=False), #DeepAR only supports distributional outputs.
    #            learning_rate=learning_rate,
    #            futr_exog_list=futr_exog_list,
    #            max_steps=training_steps,
    #            val_check_steps=val_check_steps,
    #            early_stop_patience_steps=early_stop_patience_steps,
    #            scaler_type=scaler_type,
    #            enable_progress_bar=True, 
    #            random_seed=42),

    # DeepNPTS(loss=loss,
    #              h=horizon,
    #              input_size=context_size,
    #              futr_exog_list=futr_exog_list,
    #              max_steps=training_steps,
    #              val_check_steps=val_check_steps,
    #              early_stop_patience_steps=early_stop_patience_steps,
    #              scaler_type=scaler_type,
    #              enable_progress_bar=True, 
    #              random_seed=42),
    
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
    # RNN(h=horizon,
    #         input_size=context_size,
    #         loss=DistributionLoss(distribution='Normal', return_params=False),#better with distributional
    #         # loss=loss,
    #         scaler_type=scaler_type,
    #         encoder_n_layers=2,
    #         encoder_hidden_size=128,
    #         context_size=context_size,
    #         decoder_hidden_size=128,
    #         decoder_layers=2,
    #         max_steps=training_steps,
    #         futr_exog_list=futr_exog_list,
    #         early_stop_patience_steps=early_stop_patience_steps, 
    #         random_seed=42
    #         ),
        ]
# Train the full model once on all historical data except the final forecast horizon
train_df_nf = df.groupby('unique_id').apply(lambda g: g.iloc[0:(num_train + context_size + horizon)]).reset_index(drop=True)
nf = NeuralForecast(models=models, freq='ME')
nf.fit(df=train_df_nf, val_size=horizon)

df_demand_long = df
# Create an empty DataFrame with specified columns
shap_results_group = pd.DataFrame(columns=['unique_id', 'ds','y', 'temperature_ex', 'promo_ex','DilatedRNN','DilatedRNN_Base','temperature_ex_shap','promo_ex_shap'])
accuracy_results_group = []
for unique_id in df_demand_long['unique_id'].unique().tolist():
    print('unique_id', unique_id)
    # 1. Filter for the specific unique_id if needed
    df_target = df_demand_long[df_demand_long['unique_id'] == unique_id].copy()

    # 2. Sort by date to ensure order
    df_target = df_target.sort_values('ds').reset_index(drop=True)
    df_target_test = df_target.iloc[-(12 + horizon + 9 - 1):]   # the long_zie of the testing 
    df_target_test_future = df_target.iloc[-(horizon + 9-1):]

    # 3. Select last context (12 months) and future (3 months)
    for i in list(range(num_test)):
        print('Now the current testing index is ', i)
        context_df = df_target_test.iloc[i: context_size + i]  # last 12 rows before the last 3
        future_df = df_target_test.iloc[context_size + i : context_size + i + horizon] 
        
        shap_results_eachdate = future_df.copy()
        future_df_grouped = future_df.groupby('unique_id').agg({
            'ds': list,
            'y' :list,
            'temperature_ex': list,
            'promo_ex': list,
        }).reset_index()
        future_df_grouped['target_date'] = future_df_grouped['ds'][0][2]
        future_df_grouped['y_lag2'] = future_df_grouped['y'][0][2]
        
        future_df = future_df.reset_index(drop=True).drop(columns=["y"], errors='ignore')
        
        ## get the prediction on model
        pred = nf.predict(df=context_df, futr_df=future_df)
        shap_results_eachdate['DilatedRNN'] = pred['DilatedRNN'].astype(float).round(2).values
        future_df_grouped['DilatedRNN'] = [pred['DilatedRNN'].astype(float).round(2).values.tolist()]
        future_df_grouped['DilatedRNN_pred_lag2'] = future_df_grouped['DilatedRNN'][0][2]

        ## get the expected prediction on model
        # 3. Define reference values
        ref_temperature = future_df['temperature_ex'].mean()
        ref_promo = future_df['promo_ex'].mean()
        
        df_exp = future_df.copy()
        df_exp['temperature_ex'] = ref_temperature
        df_exp['promo_ex'] = ref_promo

        pred = nf.predict(df=context_df, futr_df=df_exp)
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

                fut_S = modify_future(future_df, subset)
                fut_S_i = modify_future(future_df, subset_with_i)

                # nf.forecast(X_df=fut_S).reset_index()
                pred_S = nf.predict(df=context_df, futr_df=fut_S)
                pred_S_i = nf.predict(df=context_df, futr_df=fut_S_i)

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
         
        feature_names = ['temperature_ex', 'promo_ex']
        shap_value_names = ['temperature_ex_shap', 'promo_ex_shap']

    save_dataframe_to_json(df=shap_results_group, file_path=file_path_group_shap )
    shap_results_group.to_csv(file_path_group_shap .replace('.json', '.csv'))

    shap_results_group_filter = shap_results_group.loc[shap_results_group['unique_id'] == unique_id, ['y_lag2','DilatedRNN_pred_lag2']]
    Accuracy_adjusted_MAPE = Accuracy_adjusted_MAPE_TCCC(shap_results_group_filter['y_lag2'] , shap_results_group_filter['DilatedRNN_pred_lag2'])
    print('The Accuracy_adjusted_MAPE is :', unique_id, Accuracy_adjusted_MAPE)
    
    accuracy_results_group.append({
        'unique_id': unique_id,
        'Accuracy_adjusted_MAPE': Accuracy_adjusted_MAPE
    })

# Create DataFrame
accuracy_results_group = pd.DataFrame(accuracy_results_group)
    
save_dataframe_to_json(df=accuracy_results_group, file_path=file_path_group_accuracy )
accuracy_results_group.to_csv(file_path_group_accuracy.replace('.json', '.csv'))
 
Accuracy_adjusted_MAPE = Accuracy_adjusted_MAPE_TCCC(shap_results_group['y_lag2'] , shap_results_group['DilatedRNN_pred_lag2'])
print('The Overall Accuracy_adjusted_MAPE is :', Accuracy_adjusted_MAPE)

# level 1: shap value on each testing data
shap_results_last = shap_results_group[['unique_id']].copy()

# 遍历其余列并提取每行列表的最后一个值
for col in shap_results_group.columns:
    if col != 'unique_id':
        shap_results_last[col] = shap_results_group[col].apply(lambda x: x[-1] if isinstance(x, list) else x)


print('well done')





