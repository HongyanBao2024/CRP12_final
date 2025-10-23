import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
from einops import rearrange
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from tqdm import tqdm
# import re
# import ast
from uni2ts.eval_util.plot import plot_single, plot_next_multi
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from utils.utils_artc import calculate_mean_variability_sku_facility,string_to_number_list,calculate_rowwise_accuracy,  fill_none_with_ffill_bfill,Accuracy_adjusted_MAPE_TCCC, save_dataframe_to_json,compute_model_accuracy
import numpy as np
import torch
from einops import rearrange
import sys
import pickle

def generate_predictions(
    context_array_all,
    price_array_all,
    temperature_array_all=None,
    module_name=None,
    horizon=10,
    context_len=30,
    patch_size=5,
    num_samples=100,
    torch_manual_seed=0,
    setting=None,
):
    """
    Generate predictions (univariate or multivariate) for context arrays using the specified model.

    Parameters:
    - context_array_all: List of context arrays to process.
    - price_array_all: List of price arrays corresponding to the context arrays.
    - temperature_array_all: List of temperature arrays for multivariate forecasting (optional).
    - model_class: The forecasting model class to use.
    - module_name: Name of the pre-trained module.
    - horizon: Forecasting horizon length.
    - context_len: Context length for the model.
    - patch_size: Patch size for the model.
    - num_samples: Number of samples for prediction.
    - torch_manual_seed: Seed for reproducibility.
    - setting: List of feature names for multivariate forecasting (e.g., ['temperature', 'price']).

    Returns:
    - predictions: List of predicted values for each context array.
    """
    # Set manual seed for reproducibility
    torch.manual_seed(torch_manual_seed)
    print(f"torch_manual_seed = {torch_manual_seed}")
    
    # Determine if the task is univariate or multivariate
    is_multivariate = setting is not None and len(setting) > 0

    # Initialize model
    feat_dynamic_real_dim = len(setting) if is_multivariate else 0
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
        prediction_length=horizon,
        context_length=context_len,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=feat_dynamic_real_dim,
        past_feat_dynamic_real_dim=0,
    )
    
    # Prepare predictions list
    predictions = []

    # Process each row for prediction
    for idx, (context_array, price_array) in enumerate(zip(context_array_all, price_array_all)):
        try:
            # Ensure price array values are positive
            price_array = [abs(p) for p in price_array]
            
            # Prepare dynamic features for multivariate forecasting
            if is_multivariate:
                # Ensure temperature_array_all is provided for multivariate setting
                if temperature_array_all is None:
                    raise ValueError("Temperature data is required for multivariate forecasting.")
                
                temperature_array = temperature_array_all[idx]
                df_dict = {
                    'temperature': temperature_array,
                    'promo': price_array,
                }
                features = [df_dict[e] for e in setting]
                feat_dynamic_real = torch.as_tensor(features, dtype=torch.float32).T.unsqueeze(0)
            else:
                feat_dynamic_real = None
            
            # Prepare input tensor
            input_tensor = rearrange(torch.as_tensor(context_array, dtype=torch.float32), "t -> 1 t 1")
            observed_tensor = torch.ones_like(input_tensor, dtype=torch.bool)
            padding_tensor = torch.zeros_like(input_tensor, dtype=torch.bool).squeeze(-1)

            # Generate forecasts
            forecast = model(
                past_target=input_tensor,
                past_observed_target=observed_tensor,
                past_is_pad=padding_tensor,
                feat_dynamic_real=feat_dynamic_real,
                observed_feat_dynamic_real=feat_dynamic_real,
            )

            # Calculate the median prediction and store it
            predicted_values = np.median(forecast[0], axis=0)
            predictions.append(predicted_values.tolist())
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            predictions.append(None)  # Append None for rows that failed
            
    return predictions

def main():
    model_id = 1
    horizon =  12
    LAG =  3
    context_size = 12
    freqency = 'M'
    testing_len = 11
    task = 'CRP12'

    # identifier = f'/mnt/c/data/tccc_sg/KC_monthly/{task}_context_size_{context_size}_horizon_{horizon}_testing_len_{testing_len}_KC_Lag3_deep_robust_all'
    identifier = f'/mnt/c/data/CRP12/KC_monthly/{task}_context_size_{context_size}_horizon_{horizon}_testing_len_{testing_len}_KC_Lag3_deep_robust_all'
    file_path = f'{identifier}'
    models_acc_path = 'models_accuracy.pkl'

    
    # identifier = f'{sku_facility_domain}_context_size_{context_size}_horizon_{testing_len}_model_id_{model_id}' 
    file_path_s3 = f'{identifier}/df_all_mixing_training_s3.json'
    file_path_s4 = f'{identifier}/df_all_mixing_training_s4.json'
    file_path_s4_temp = f'{identifier}/df_all_mixing_training_s4_temp.json'
    patch_size = 16  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
    num_samples = 100
    torch_manual_seed = 0

    df_all = pd.read_json(file_path_s3, orient='records', lines=True)
    
    if model_id == 1:
        print(f"=========================model_id = {model_id}, making the one time UNI2TS forecasting=========================")
        context_array_all = df_all['context'].to_list()
        context_array_all = [string_to_number_list(e) if isinstance(e, str) else e for e in context_array_all]

        temperature_array_all = df_all['temperature_ex'].to_list()
        temperature_array_all = [string_to_number_list(e) if isinstance(e, str) else e for e in temperature_array_all]
    
        price_array_all = df_all['promo_ex'].to_list()
        price_array_all = [string_to_number_list(e) if isinstance(e, str) else e for e in price_array_all]

        # uni_variate
        predictions_uni = generate_predictions(
            context_array_all=context_array_all,
            price_array_all=price_array_all,
            module_name="Salesforce/moirai-1.0-R-small",
            horizon=horizon,
            context_len=context_size,
            patch_size=patch_size,
            num_samples=num_samples,
            torch_manual_seed=torch_manual_seed,
            setting=None,  # No additional features for univariate
        )


        # External factors
        predictions_ex = generate_predictions(
            context_array_all=context_array_all,
            price_array_all=price_array_all,
            temperature_array_all=temperature_array_all,
            module_name="Salesforce/moirai-1.0-R-small",
            horizon=horizon,
            context_len=context_size,
            patch_size=patch_size,
            num_samples=num_samples,
            torch_manual_seed=torch_manual_seed,
            setting=['temperature', 'promo'],  # Features for multivariate
        )
        
        # save resutls by mergin gwith previous one
        df_all['uni2ts'] = predictions_uni
        df_all['uni2ts_ex'] = predictions_ex
    else:
        print(f"=========================model_id = {model_id} , loaded from temp=========================")
        df_s4 = pd.read_json(file_path_s4_temp, orient='records', lines=True)
        df_all['uni2ts'] = df_s4['uni2ts']
        df_all['uni2ts_ex'] = df_s4['uni2ts_ex']
        
    save_dataframe_to_json(df=df_all, file_path=file_path_s4)
    df_all.to_csv(file_path_s4.replace('.json', '.csv'))

    model_start_df = 4
    file_path_mixing_training_s4 = file_path_s4
    print(f"file_path_mixing_training_s4 is {file_path_mixing_training_s4}")
    df = pd.read_json(file_path_mixing_training_s4,
                      orient='records', lines=True)

    model_start_df = 7

    seed_np = 42
    np.random.seed(seed_np)
    df_all['LAG3_true'] = df_all.sales_N.apply(lambda x: x[LAG-1])
    y_true = df_all['LAG3_true'].to_numpy().reshape(-1)
    results = []
    for model_index in range(model_start_df,len(df.columns)):
        model_name = df.columns[model_index]
        pred_name = 'LAG3_pred' + '_' + model_name
        df_all[pred_name] = df_all.iloc[:,model_index].apply(lambda x: x[LAG-1])

        y_pred = df_all[pred_name].to_numpy().reshape(-1)
        acc_list = calculate_rowwise_accuracy(y_true, y_pred)
        df_all['acc_' + pred_name] = acc_list
        WFA_model = Accuracy_adjusted_MAPE_TCCC(y_true=y_true, y_pred= y_pred)
        results.append(WFA_model)

    df_accuracy = pd.DataFrame(results, index= df.columns[model_start_df:len(df.columns)], columns=['WFA'])
    df_sorted = df_accuracy.sort_values('WFA', ascending=False).T
        
    save_dataframe_to_json(df=df_all, file_path=file_path_s4)
    df_all.to_csv(file_path_s4.replace('.json', '.csv'))

    #save used model pool
    with open(os.path.join(file_path, models_acc_path), "wb") as f:
        pickle.dump(df_accuracy, f)

    df_accuracy['WFA (%)'] = df_accuracy['WFA'] * 100

    # Plot the bar chart
    plt.figure(figsize=(15, 6))
    bars = plt.bar(df_accuracy.index,
                    df_accuracy['WFA (%)'], color='skyblue', edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.1f}%',  # Format as a percentage with 2 decimal places
            ha='center',
            va='bottom',
            fontsize=6
        )
    plt.tight_layout()
    # Customize the plot
    plt.title('Adjusted_MAPE Accuracy (%) per Model', fontsize=7)
    plt.ylabel('Adjusted_MAPE Accuracy (%)', fontsize=5)
    plt.xlabel('Models', fontsize=5)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    models_acc_path_ = os.path.join(file_path, models_acc_path)
    plt.tight_layout()
    plt.savefig(f'{models_acc_path_.replace(".pkl", "_bar_chart_allsales_all_smooth.png")}', dpi=300)
    

if __name__ == "__main__":
    main()

