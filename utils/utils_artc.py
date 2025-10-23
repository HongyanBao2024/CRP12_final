import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
import numpy as np
import torch
import logging


def filter_rows_sku_fac(df):
    return df[df.unique_id.str.contains('-')]
# Suppose you want to select columns that end with a digit after a hyphen
def filter_columns_sku_fac(df: pd.DataFrame) -> pd.DataFrame:
    pattern = '-' 
    mask = df.columns.astype(str).str.contains(pattern, regex=True, na=False)
    selected_columns = df.columns.astype(str)[mask]
    return df[selected_columns].copy()

def get_df_context_df_future_univari_weekly(df, rolling_id, horizon, LAG, context_size, testing_len=None):
    if testing_len is None:
        testing_len = horizon
    results_context = []
    results_future = []
    # Iterate over each row in the original DataFrame
    for idx, row in df.iterrows():
        if idx%(testing_len-LAG+1) == rolling_id:
            sku = row['SKU_facility']
            ## sales data context
            sales_data_context = row['context']
            
            ## dates context
            dates_context = row['data_range']

            dates_future = pd.date_range(start= dates_context[-1] + pd.DateOffset(weeks=1), periods=horizon, freq='W-FRI')  # Adjust the start date accordingly
            
            temp_df_context = pd.DataFrame({
                'unique_id': sku,
                'ds': dates_context,
                'y': sales_data_context,
            })
            results_context.append(temp_df_context)
            
            temp_df = pd.DataFrame({
                'unique_id': sku,
                'ds': dates_future,
            }) # predict the y information
            
            results_future.append(temp_df)

    # Concatenate all the individual DataFrames into one final DataFrame
    df_context= pd.concat(results_context, ignore_index=True)
    df_future = pd.concat(results_future, ignore_index=True)

    # for idx, row in df.iterrows():
    #     context = row['context']
    #     date_range = row['context_date']
    #     sku = row['category']
    #     df_new = pd.DataFrame({
    #         'unique_id': sku,
    #         'ds': date_range,
    #         'y': context,
    #     })
    #     results.append(df_new)

    # df_new = pd.concat(results, ignore_index=True)
    # df_new['ds'] = pd.to_datetime(df_new['ds'] )
    # ### make df_futr
    # df_future = nf.make_future_dataframe(df_new)

    # y = nf.predict(df=df_new,futr_df=df_future)
    # y_all.append(y)
    return df_context, df_future

def get_df_context_df_future_univari(df, rolling_id, horizon, LAG, context_size, testing_len=None):
    if testing_len is None:
        testing_len = horizon
    results_context = []
    results_future = []
    # Iterate over each row in the original DataFrame
    for idx, row in df.iterrows():
        if idx%(testing_len-LAG+1) == rolling_id:
            sku = row['SKU_facility']
            ## sales data context
            sales_data_context = row['context']
            
            ## dates context
            dates_context = row['data_range']

            dates_future = pd.date_range(start= dates_context[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq='M')  # Adjust the start date accordingly
            
            temp_df_context = pd.DataFrame({
                'unique_id': sku,
                'ds': dates_context,
                'y': sales_data_context,
            })
            results_context.append(temp_df_context)
            
            temp_df = pd.DataFrame({
                'unique_id': sku,
                'ds': dates_future,
            }) # predict the y information
            
            results_future.append(temp_df)

    # Concatenate all the individual DataFrames into one final DataFrame
    df_context= pd.concat(results_context, ignore_index=True)
    df_future = pd.concat(results_future, ignore_index=True)

    # for idx, row in df.iterrows():
    #     context = row['context']
    #     date_range = row['context_date']
    #     sku = row['category']
    #     df_new = pd.DataFrame({
    #         'unique_id': sku,
    #         'ds': date_range,
    #         'y': context,
    #     })
    #     results.append(df_new)

    # df_new = pd.concat(results, ignore_index=True)
    # df_new['ds'] = pd.to_datetime(df_new['ds'] )
    # ### make df_futr
    # df_future = nf.make_future_dataframe(df_new)

    # y = nf.predict(df=df_new,futr_df=df_future)
    # y_all.append(y)
    return df_context, df_future

def get_df_context_df_future(df, rolling_id, horizon, LAG, context_size, testing_len=None):
    if testing_len is None:
        testing_len = horizon
    results_context = []
    results_future = []
    # Iterate over each row in the original DataFrame
    for idx, row in df.iterrows():
        if idx%(testing_len-LAG+1) == rolling_id:
            print(idx)
            sku = row['unique_id']
            ## sales data context
            sales_data_context = row['context']
            
            ## dates context
            dates_context = row['data_range']
            
            ## temperature context
            temperature_data = row['temperature_ex']
            temperature_data_context = temperature_data[:context_size]
            temperature_data_ex = temperature_data[-horizon:]
            
            ## price context
            price_data = row['promo_ex']
            price_data_context = price_data[:context_size]
            price_data_ex = price_data[-horizon:]
            # dates_future = pd.date_range(start='2024-03-10', periods=horizon, freq=freq)  # Adjust the start date accordingly
            dates_future = row['data_range_ex'][-horizon:]
            
            temp_df_context = pd.DataFrame({
                'unique_id': sku,
                'ds': dates_context,
                'y': sales_data_context,
                'temperature_ex': temperature_data_context,
                'promo_ex': price_data_context
            })
            results_context.append(temp_df_context)
            
            temp_df = pd.DataFrame({
                'unique_id': sku,
                'ds': dates_future,
                'temperature_ex': temperature_data_ex,
                'promo_ex': price_data_ex
            }) # predict the y information
            
            results_future.append(temp_df)

    # Concatenate all the individual DataFrames into one final DataFrame
    # for model testing
    df_context= pd.concat(results_context, ignore_index=True)
    df_future = pd.concat(results_future, ignore_index=True)

    return df_context, df_future

def get_df_context_df_future_shap(df, horizon, context_size, testing_len=None):
    if testing_len is None:
        testing_len = horizon
    results_context = []
    results_future = []
    # Iterate over each row in the original DataFrame
    for idx, row in df.iterrows():
        sku = row['unique_id']
        ## sales data context
        sales_data_context = row['context']
        
        ## dates context
        dates_context = row['data_range']
        
        ## temperature context
        temperature_data = row['temperature_ex']
        temperature_data_context = temperature_data[:context_size]
        temperature_data_ex = temperature_data[-horizon:]
        
        ## price context
        price_data = row['promo_ex']
        price_data_context = price_data[:context_size]
        price_data_ex = price_data[-horizon:]
        # dates_future = pd.date_range(start='2024-03-10', periods=horizon, freq=freq)  # Adjust the start date accordingly
        dates_future = row['data_range_ex'][-horizon:]
        
        temp_df_context = pd.DataFrame({
            'unique_id': sku,
            'ds': dates_context,
            'y': sales_data_context,
            'temperature_ex': temperature_data_context,
            'promo_ex': price_data_context
        })
        results_context.append(temp_df_context)
        
        temp_df = pd.DataFrame({
            'unique_id': sku,
            'ds': dates_future,
            'temperature_ex': temperature_data_ex,
            'promo_ex': price_data_ex
        }) # predict the y information
        
        results_future.append(temp_df)

    # Concatenate all the individual DataFrames into one final DataFrame
    # for shap value 
    df_context=  pd.DataFrame(results_context[0], columns=["unique_id", "ds", "y", "temperature_ex", "promo_ex"])
    df_future =  pd.DataFrame(results_future[0], columns=["unique_id", "ds", "y", "temperature_ex", "promo_ex"])
    # for model testing
    # df_context= pd.concat(results_context, ignore_index=True)
    # df_future = pd.concat(results_future, ignore_index=True)
    return df_context, df_future

def select_rows_by_date_range(df, start_date, end_date):
    """
    Selects rows within a specified weekly date range from the DataFrame.
    If the DataFrame doesn't have enough data to cover the date range,
    fills missing dates using data from the same date 53 weeks earlier.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with a datetime index at weekly frequency.
    - start_date (str or pd.Timestamp): The start date for filtering (inclusive).
    - end_date (str or pd.Timestamp): The end date for filtering (inclusive).

    Returns:
    - pd.DataFrame: Filtered and extended DataFrame covering the entire date range.
    """
    # Step 1: Validate Input DataFrame
    if df.empty:
        raise ValueError("Input DataFrame 'df' is empty. Cannot fill missing dates with data from 53 weeks earlier.")
    
    # Step 2: Ensure the index is in datetime format
    df = df.copy()  # To avoid modifying the original DataFrame
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    
    # Step 3: Sort the DataFrame by index to ensure chronological order
    df = df.sort_index()
    
    # Step 4: Convert start_date and end_date to Timestamps
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Step 5: Validate date range
    if start_date > end_date:
        raise ValueError("start_date must be earlier than or equal to end_date.")
    
    # Step 6: Infer the frequency of the DataFrame's index
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq is None:
        raise ValueError("Cannot infer frequency of the DataFrame's index. Please ensure it has a regular weekly frequency (e.g., 'W-MON').")
    # logging.info(f"Inferred frequency: {inferred_freq}")
    
    # Step 7: Create a complete date range from start_date to end_date
    full_date_range = pd.date_range(start=start_date, end=end_date, freq=inferred_freq)
    # logging.info(f"Full date range created from {start_date.date()} to {end_date.date()}.")
    
    # Step 8: Filter the DataFrame for the specified date range
    filtered_df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    # logging.info(f"Initial filtered data has {len(filtered_df)} rows.")
    
    # Step 9: Identify missing dates within the date range
    missing_dates = full_date_range.difference(filtered_df.index)
    # logging.info(f"Number of missing dates to fill: {len(missing_dates)}")
    
    if missing_dates.empty:
        # If no dates are missing, return the filtered DataFrame
        # logging.info("No missing dates found. Returning the filtered DataFrame.")
        return filtered_df
    
    else:
        # Initialize a list to hold DataFrames for missing dates
        fill_dfs = []
        
        for missing_date in missing_dates:
            # Calculate the corresponding date 52 weeks earlier
            try:
                earlier_date = missing_date - pd.offsets.MonthEnd(12)
            except Exception as e:
                # logging.error(f"Error calculating date 52 weeks back for {missing_date.date()}: {e}")
                continue  # Skip this missing date
            
            # Attempt to retrieve data for earlier_date
            if earlier_date in df.index:
                # Use double brackets to ensure a DataFrame is returned
                fill_row = df.loc[[earlier_date]]
                
                # Assign the missing_date to the index
                fill_row.index = [missing_date]
                
                fill_dfs.append(fill_row)
                # logging.info(f"Filled {missing_date.date()} using data from {earlier_date.date()}.")
            else:
                # If data from 52 weeks earlier is not available, fill with NaN
                # logging.warning(f"Data from {earlier_date.date()} not found. Filling {missing_date.date()} with NaN.")
                empty_row = pd.DataFrame({col: [pd.NA] for col in df.columns}, index=[missing_date])
                fill_dfs.append(empty_row)
        
        if fill_dfs:
            # Concatenate all filled rows
            filled_data = pd.concat(fill_dfs)
            
            # Combine with the originally filtered data
            combined_df = pd.concat([filtered_df, filled_data]).sort_index()
            
            # logging.info(f"Combined DataFrame has {len(combined_df)} rows.")
            
            return combined_df
        else:
            # If no filling was possible, return the filtered DataFrame as is
            # logging.warning("No missing dates could be filled with data from 53 weeks earlier.")
            return filtered_df

def get_nf_model_names(nf):
    # add _ex if there are two (meaning external factors)
    models = [str(name )for name in nf.models]
    # Initialize an empty list to store the result
    result = []
    # Initialize a dictionary to keep track of the occurrence of each model
    model_count = {}
    # Loop through the list and append "1" if it's not the first occurrence
    for name in models:
        if name in model_count:
            model_count[name] += 1
            result.append(f"{name}1")
        else:
            model_count[name] = 1
            result.append(name)

    # Output the result
    model_names = result
    return model_names

def extract_forecasting_res(y_all, model_names, horizon):
    """
    Extracts forecasting results from multiple models and consolidates them into a single DataFrame.
    
    Parameters:
    - y_all (list of pd.DataFrame): List of DataFrames containing forecasting results.
      Each DataFrame should have a 'ds' column for dates and columns corresponding to each model in model_names.
      The index of each DataFrame should represent unique SKUs.
    - model_names (list of str): List of model names corresponding to columns in the DataFrames.
    - horizon (int): Number of weeks to forecast (i.e., number of future weeks to extract).
    
    Returns:
    - pd.DataFrame: Consolidated DataFrame with predictions from all models.
    """
    # Initialize list to store predictions for each DataFrame in y_all
    pred_all_rolling_ids = []
    
    for y in y_all:
        all_df_pred = []
        for model_name in model_names:
            np_wide_all = []
            sku_all = []
            
            # Create a copy to avoid modifying the original DataFrame
            y_processed = y.copy()
            
            # Assign 'unique_id' as the SKU identifier from the index
            if 'unique_id'  not in y_processed.columns: 
                y_processed['unique_id'] = y_processed.index
            
            # Pivot the DataFrame to have 'unique_id' as rows and 'ds' as columns
            df_wide = y_processed.pivot(index='unique_id', columns='ds', values=model_name)
            df_wide.reset_index(inplace=True)
            
            # Extract the last 'horizon' weeks of data
            forecast_values = df_wide.iloc[:, -horizon:].values
            np_wide_all.append(forecast_values)
            
            # Collect SKUs
            sku_all += df_wide['unique_id'].tolist()
            
            # Stack predictions vertically (if multiple iterations were present)
            np_wide_all = np.vstack(np_wide_all)
            
            # Create a DataFrame with 'SKU' and predictions
            df_result = pd.DataFrame({
                'SKU': sku_all,
                f'pred_{model_name}': list(np_wide_all)
            })
            
            # Add an auxiliary index to maintain original order
            df_result['original_index'] = df_result.index
            
            # Sort by 'SKU' and then by 'original_index' to maintain sequence within each SKU
            df_result_sorted = df_result.sort_values(by=['SKU', 'original_index']).reset_index(drop=True)
            
            # Drop the auxiliary index column
            df_result_sorted = df_result_sorted.drop(columns=['original_index'])
            
            # Append the sorted DataFrame to the list of predictions
            all_df_pred.append(df_result_sorted)
        
        # Append all model predictions for the current DataFrame to the main list
        pred_all_rolling_ids.append(all_df_pred)
    
    # Concatenate predictions across models for each DataFrame in y_all
    concat_list = [pd.concat(pred, axis=1).drop(columns=["SKU"], errors='ignore') for pred in pred_all_rolling_ids]
    
    # Initialize a list to collect all rows
    consolidated_rows = []
    
    # Assuming all DataFrames in concat_list have the same number of rows
    num_rows_per_df = len(concat_list[0])
    
    # Iterate over each row index and collect corresponding rows from each DataFrame
    for row_index in range(num_rows_per_df):
        for df in concat_list:
            consolidated_rows.append(df.iloc[row_index])
    
    # Concatenate all collected rows into a single DataFrame
    df_all = pd.concat(consolidated_rows, ignore_index=True)
    
    # Reshape the DataFrame to have one row per prediction set
    reshaped_array = np.array(df_all).reshape(-1, len(model_names)).tolist()
    
    # Modify model names to ensure uniqueness and clarity
    new_model_names = []
    for model_name in model_names:
        if model_name.endswith("1"):
            model_name = model_name[:-1] + "_ex"
        new_model_names.append(model_name)
    
    # Ensure model names are unique by appending counts to duplicates
    name_count = {}
    unique_model_names = []
    for name in new_model_names:
        if name in name_count:
            name_count[name] += 1
            unique_model_names.append(f"{name}_{name_count[name]}")
        else:
            name_count[name] = 1
            unique_model_names.append(name)
    
    # Create the final DataFrame with consolidated predictions
    df_pred_all = pd.DataFrame(reshaped_array, columns=unique_model_names)
    
    return df_pred_all

def process_long_data(df, freq='M', index='date', sku_column='sku_code', target_column='weighted_avg_unit_price_discount'):
    # Load the data
    df_temp = df.pivot(index=index, columns=sku_column, values=target_column)
    # Determine columns based on data type
    if freq == 'W-FRI' or freq == 'w':
        df_temp.index = pd.to_datetime(df_temp.index).to_period('W-FRI').to_timestamp('W-FRI')
    elif freq == 'M' or freq == 'm':
        df_temp.index = pd.to_datetime(df_temp.index).to_period(freq).to_timestamp(freq)
        # df_temp.index = pd.to_datetime(df_temp.index).to_period('M').to_timestamp('M') 
    # Sort the rows based on the index and handle missing values
    df_temp.sort_index(inplace=True)  
    # <bound method Index._wrap_setop_result of DatetimeIndex(
            #    ['2020-01-19', '2020-01-26', '2020-02-02', '2020-02-09',
            #    '2020-02-16', '2020-02-23', '2020-03-01', '2020-03-08',
            #    '2020-03-15', '2020-03-22',
            #    ...
            #    '2024-05-12', '2024-05-19', '2024-05-26', '2024-06-02',
            #    '2024-06-09', '2024-06-16', '2024-06-23', '2024-06-30',
            #    '2024-07-07', '2024-07-14'],
            #   dtype='datetime64[ns]', name='ds', length=235, freq='W-FRI')>
    # df_temp.fillna(method='ffill', axis=0, inplace=True)
    # df_temp.fillna(method='bfill', axis=0, inplace=True)
    df_temp.ffill(axis=0)
    df_temp.bfill(axis=0)

    return df_temp

def mask_demask_unique_id(df, mask=True):
    """
    Masks or demasks the 'unique_id' column in a DataFrame by adding or removing the '_artc' suffix.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the 'unique_id' column.
    mask : bool, optional (default=True)
        If True, masks the 'unique_id' by appending '_artc'.
        If False, demasks by removing the '_artc' suffix.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with the 'unique_id' column masked or demasked.
    
    Raises:
    -------
    ValueError:
        If the 'unique_id' column is not present in the DataFrame.
    """
    # Check if 'unique_id' column exists
    if 'unique_id' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'unique_id' column.")
    
    # Create a copy to avoid modifying the original DataFrame
    df_modified = df.copy()
    
    # Handle NaN values by keeping them as NaN
    unique_ids = df_modified['unique_id'].astype(str)
    
    if mask:
        # Masking: Append '_artc' if not already present
        df_modified['unique_id'] = unique_ids.apply(
            lambda x: x if x.endswith('_artc') else f"{x}_artc" if x.lower() != 'nan' else np.nan
        )
    else:
        # Demasking: Remove '_artc' if present at the end
        df_modified['unique_id'] = unique_ids.str.replace('_artc$', '', regex=True)
        # Convert 'nan' strings back to actual NaN values
        df_modified['unique_id'] = df_modified['unique_id'].replace('nan', np.nan)
    
    return df_modified

def _add_weather_data_with_unknown_facility(demand_KBL, weather_data_path):
    '''Integrates weather data with the main demand dataset'''
    demand_KBL['facility'] = demand_KBL['unique_id'].str.split('-').str[1]
    demand_KBL['ds'] = pd.to_datetime(demand_KBL['ds'])
    weather_data = pd.read_csv(weather_data_path)
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    #average T based on dates without considering the SKU difference
    average_T = weather_data[['date', 'temperature_max', 'temperature_average','temperature_min', 'precipitation']].groupby('date').mean()
    average_T.reset_index(inplace=True)
    #missing facilities
    weather_facility = weather_data['facility'].unique().tolist()
    kbl_facility = demand_KBL['facility'].unique().tolist()
    missing_facility_weather = [a for a in kbl_facility if a not in weather_facility]
    # append missing facility using average data
    facility_weather_dfs = []
    print(missing_facility_weather)
    for fac in missing_facility_weather:
        new_df = average_T.copy()
        new_df['facility'] = fac
        facility_weather_dfs.append(new_df)
    df = pd.concat(facility_weather_dfs, axis=0)
    df = df[['facility', 'date', 'temperature_max', 'temperature_average', 'temperature_min', 'precipitation']]
    new_weather_data = pd.concat([weather_data, df], axis=0)
    combined_data = pd.merge(demand_KBL, new_weather_data, left_on=['facility', 'ds'], right_on=['facility', 'date'], how='left')
    combined_data.reset_index(inplace=True)
    return combined_data

def make_MLforecast_with_context_and_future(mlf, df_context, df_future, external_factor=True, horizon=13):
    if external_factor:
        df_context_ML = mask_demask_unique_id(df_context, mask=True)
        df_future_ML = mask_demask_unique_id(df_future, mask=True)
        forecast = mlf.predict(h=horizon, new_df=df_context_ML,X_df=df_future_ML)
        forecast = mask_demask_unique_id(forecast, mask=False)
    else:
        forecast = mlf.predict(h=horizon, new_df=df_context[['unique_id', 'ds', 'y']])
    return forecast

def prepare_training_data(
    df_train_sales, 
    df_temperature, 
    df_price, 
    training_date_range
):
    """
    Processes training data by iterating over sales data and combining it with temperature and price data.

    Parameters:
    ----------
    df_train_sales : pd.DataFrame
        DataFrame containing sales data for different SKU-facility combinations.
    df_temperature : pd.DataFrame
        DataFrame containing temperature data for different SKU-facility combinations.
    df_price : pd.DataFrame
        DataFrame containing price data for different SKU-facility combinations.
    training_date_range : pd.DatetimeIndex or list
        A range of dates corresponding to the training period.

    Returns:
    -------
    pd.DataFrame
        A concatenated DataFrame combining sales, temperature, and price data for all SKU-facility combinations.
    """
    results = []
    
    # Iterate over each row in the sales DataFrame
    for idx, row in df_train_sales.iterrows():
        sku_facility = row.name
        sales_data = row.values.tolist()
        num_sales = len(sales_data)
        
        # Match the length of temperature and price data to sales data
        temperature_data = df_temperature[sku_facility].values.tolist()[:num_sales]
        price_data = df_price[sku_facility].values.tolist()[:num_sales]
        dates = training_date_range

        # Create a temporary DataFrame for the current SKU-facility
        temp_df = pd.DataFrame({
            'unique_id': sku_facility,
            'ds': dates,
            'y': sales_data,
            'temperature_ex': temperature_data,
            'promo_ex': price_data
        })

        results.append(temp_df)
    
    # Concatenate all temporary DataFrames into one final DataFrame
    final_df = pd.concat(results, ignore_index=True)
    
    return final_df

def prepare_weekly_inputs(
    df_sales,
    testing_context_start_date_ori,
    horizon,
    LAG,
    context_size,
    testing_len = None,
    freq = 'ME',
):
    

    # Initialize lists to store results
    if testing_len is None:
        testing_len = horizon
        
    sku_fac_list, data_range_list, data_range_ex_list = [], [], []
    temperature_ex_list, price_ex_list, context_list, sales_N_list = [], [], [], []

    # Iterate over each SKU-facility in the sales DataFrame
    for idx, row in df_sales.iterrows():
        sku_facility = row.name
        
        for rolling_id in range(testing_len - LAG + 1):
            sku_fac_list.append(sku_facility)
            
            # Calculate the start date for the rolling window
            start = pd.to_datetime(testing_context_start_date_ori) + pd.DateOffset(weeks=rolling_id)
            
            # Historical context date range
            date_range = pd.date_range(start=start, periods=context_size, freq = freq)
            data_range_list.append(date_range)
            
            # Sales data for the context
            w1, w2 = date_range[0], date_range[-1]
            # print("w1,w2 for future",w1,w2)
            sales_data_context = row.loc[w1:w2].to_list()
            context_list.append(sales_data_context)
            
            # Future sales data (sales_N)
            w1, w2,w3 = pd.to_datetime(date_range[-1]) + pd.DateOffset(weeks=1),pd.to_datetime(date_range[-1]) + pd.DateOffset(weeks=2), pd.to_datetime(date_range[-1]) + pd.DateOffset(weeks=4)
            # print("w1,w2 for future",w1,w2)
            sales_N = row.loc[w1:w3].to_list()
            # print("w1,w2 for future",w1,w2,sales_N)
            sales_N_list.append(sales_N)
            
            # Extended date range for multivariate inputs
            date_range_mv = pd.date_range(start=start, periods=context_size + horizon, freq=freq)
            data_range_ex_list.append(date_range_mv)
    
    # Combine all the collected data into a single DataFrame
    df_all_inputs_weekly = pd.DataFrame({
        "SKU_facility": sku_fac_list,
        "data_range": data_range_list,
        "context": context_list,
        "sales_N": sales_N_list
    })
    
    return df_all_inputs_weekly

def prepare_monthly_inputs(
    df_sales,
    testing_context_start_date_ori,
    horizon,
    LAG,
    context_size,
    testing_len = None,
    freq = 'ME',
):
    # Initialize lists to store results
    if testing_len is None:
        testing_len = horizon
        
    sku_fac_list, data_range_list, data_range_ex_list = [], [], []
    temperature_ex_list, price_ex_list, context_list, sales_N_list = [], [], [], []

    # Iterate over each SKU-facility in the sales DataFrame
    for idx, row in df_sales.iterrows():
        sku_facility = row.name
        
        for rolling_id in range(testing_len - LAG + 1):
            sku_fac_list.append(sku_facility)
            
            # Calculate the start date for the rolling window
            start = pd.to_datetime(testing_context_start_date_ori) + pd.DateOffset(months=rolling_id)
            
            # Historical context date range
            date_range = pd.date_range(start=start, periods=context_size, freq = freq)
            data_range_list.append(date_range)
            
            # Sales data for the context
            w1, w2 = date_range[0], date_range[-1]
            # print("w1,w2 for future",w1,w2)
            sales_data_context = row.loc[w1:w2].to_list()
            context_list.append(sales_data_context)
            
            # Future sales data (sales_N)
            w1, w2,w3 = pd.to_datetime(date_range[-1]) + pd.offsets.MonthEnd(1),pd.to_datetime(date_range[-1]) + pd.offsets.MonthEnd(2), pd.to_datetime(date_range[-1]) + pd.offsets.MonthEnd(3)
            # print("w1,w2 for future",w1,w2)
            sales_N = row.loc[w1:w3].to_list()
            # print("w1,w2 for future",w1,w2,sales_N)
            sales_N_list.append(sales_N)
            
            # Extended date range for multivariate inputs
            date_range_mv = pd.date_range(start=start, periods=context_size + horizon, freq=freq)
            data_range_ex_list.append(date_range_mv)
            

    
    # Combine all the collected data into a single DataFrame
    df_all_inputs_monthly = pd.DataFrame({
        "SKU_facility": sku_fac_list,
        "data_range": data_range_list,
        "context": context_list,
        "sales_N": sales_N_list
    })
    
    return df_all_inputs_monthly

# Genetic algorithm function now takes the fitness evaluation function as a parameter
def genetic_algorithm_integrated_with_MLP(initial_guess, pred_models_all, df,
                      fitness_func,
                      population_size=100, 
                      num_generations=50,
                      mutation_rate=0.1,
                      penetration_ratio=0.1,
                      early_stopping_generations=50,
                      random_seed=2024,
                      outlier_fraction=0.05,
                      reproduction_rate = 0.5,
                      prob_crossover=0.8,
                      prob_uniform_mutation = 0.1,
                      use_gpu=True, 
                      true_col='sales_3', LAG=3):
    # Set the random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    num_genes = len(pred_models_all)

    # Step 1: Initialize the population
    population = [create_chromosome(num_genes) for _ in range(population_size)]
    num_guess = int(population_size*penetration_ratio)
    population[:num_guess] = [initial_guess]* num_guess

    # Track the best chromosome
    best_chromosome = None
    best_fitness = float('-inf')
    generations_without_improvement = 0  # To track convergence
    evolution_history=[]

    # Step 2: Evolution over generations
    for generation in range(num_generations):
        print(f"Generation = {generation}")
        # Step 3: Evaluate fitness of each chromosome in the population
        fitness_scores = [(chromosome, fitness_func(select_models_from_chromosome(pred_models_all, chromosome), df, use_gpu=use_gpu, true_col=true_col, LAG=LAG)) for chromosome in population]

        # Step 4: Select the best chromosome in the current generation
        current_best_chromosome, current_best_fitness = max(fitness_scores, key=lambda x: x[1])

        # Check if there is an improvement
        if current_best_fitness > best_fitness:
            best_chromosome, best_fitness = current_best_chromosome, current_best_fitness
            generations_without_improvement = 0  # Reset the counter
        else:
            generations_without_improvement += 1

        # Early stopping check
        if generations_without_improvement >= early_stopping_generations:
            print(f"Early stopping at generation {generation + 1} due to no improvement for {early_stopping_generations} generations.")
            break

        # Step 5: Select parents based on fitness (Roulette wheel selection) 
        num_parents = int(population_size * reproduction_rate)
        parent_indices = random.sample(range(population_size), num_parents)
        non_parent_indices = [i for i in range(population_size) if i not in parent_indices]

        # Separate parents and non-parents
        parents = [population[i] for i in parent_indices]
        non_parents = [population[i] for i in non_parent_indices]
        new_population = non_parents

        # Generate offspring and select the best candidates
        for i in range(0, num_parents, 2):
            prob = np.random.random()
            parent1, parent2 = parents[i], parents[(i - 1) % num_parents]  # Use modulo to handle circular indexing

            if prob < prob_crossover:
                offspring1, offspring2 = crossover_chromosomes(parent1, parent2)
            elif prob >= prob_crossover and prob < prob_crossover+prob_uniform_mutation:
                offspring1 = mutate_chromosome(parent1, mutation_rate)
                offspring2 = mutate_chromosome(parent2, mutation_rate)
            else:
                offspring1 = mutate_chromosome_point(parent1)
                offspring2 = mutate_chromosome_point(parent2)

            # List of all candidates
            all_candidates = [parent1, parent2, offspring1, offspring2]

            # Calculate or retrieve fitness for each candidate
            all_candidates_fitness = [(chromosome, fitness_func(select_models_from_chromosome(pred_models_all, chromosome), df, use_gpu=use_gpu, true_col=true_col, LAG=LAG)) for chromosome in all_candidates]

            # Sort candidates by fitness in descending order
            sorted_candidates = sorted(all_candidates_fitness, key=lambda x: x[1], reverse=True)

            # Select the top 2 chromosomes based on fitness
            top_chromosome1, top_chromosome2 = sorted_candidates[0][0], sorted_candidates[1][0]
            new_population += [top_chromosome1, top_chromosome2]

        population = new_population

        # Print the progress
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
        evolution_history.append(best_fitness)

    # Step 8: Return the best chromosome found
    return best_chromosome, best_fitness, evolution_history, population

def fitness_evaluation_with_mlp_pytorch(selected_models, df, seed_np=42, seed_torch=42, use_gpu=True, true_col='sales_3', LAG=3):
    np.random.seed(seed_np)
    torch.manual_seed(seed_torch)

    # Check if GPU is available and use_gpu is True
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

    pred_results_3m_all = []

    for c in selected_models:
        c_array = np.array([row for row in df[c].values])
        # use 2:3 to extract lag 3 mont
        pred_results_3m_all.append(c_array[:, LAG-1:LAG].round(0))

    X_all = np.hstack(pred_results_3m_all)
    Y_all = np.array([row for row in df[true_col].values])[:, (LAG-1):LAG].round(0)


    X, X_test, Y, Y_test = train_test_split(X_all, Y_all, test_size=0.4, random_state=seed_np)
        
    # Convert training data to PyTorch tensors and move to the appropriate device
    X_train = torch.from_numpy(X).float().to(device)
    Y_train = torch.from_numpy(Y).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    Y_test_tensor = torch.from_numpy(Y_test).float().to(device)

    # Initialize the MLP model, optimizer, and learning rate
    model = SimpleMLP(input_size=X.shape[1], hidden_size=(50, 60), output_size=1).to(device)
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1000):
        model.train()
        Y_pred = model(X_train)

        # Compute custom loss
        loss = amape_loss(Y_pred, Y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 100 epochs
        # if epoch % 20 == 0:
        #     print(f'Epoch [{epoch}/1000], Loss: {loss.item():.4f}')

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for evaluation/testing
    with torch.no_grad():
        # Forward pass on the test data
        y_pred = model(X_test_tensor)
        
        # Calculate MAPE loss on the test set
        test_loss = wfa_loss_TCCC(y_pred=y_pred, y_true=Y_test_tensor)

        # Calculate the fitness
        fitness = 1 - test_loss.item()

    return fitness

# Custom sMAPE loss function that removes top k outliers
def wfa_loss_TCCC(y_pred, y_true):
    # Replace negative values in predictions with 0
    y_pred = torch.clamp(y_pred, min=0)

    # Calculate the numerator: sum of absolute differences between actual and predicted values
    numerator = torch.sum(torch.abs(y_true - y_pred))
    
    # Calculate the denominator: sum of the actual and predicted values, scaled by 1/2
    denominator = torch.sum(abs(y_true))
    
    # Calculate the WFA loss
    wfa_value = 1 - (numerator / denominator)
    
    # Apply max condition to ensure non-negativity
    wfa_value = torch.clamp(wfa_value, min=0)
    
    return 1-wfa_value


def process_df_with_model(df, model):
    """
    Processes a DataFrame where each column contains lists of 13 values. 
    Creates a new DataFrame where each row is processed by the model.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with columns containing lists of 13 values.
        model (torch.nn.Module): PyTorch model to process the data.
        
    Returns:
        pd.DataFrame: New DataFrame with a 'ARTC_model' column containing processed lists.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Determine the device the model is on
    device = next(model.parameters()).device

    def process_row(row):
        # Stack the values into a tensor
        input_tensor = torch.tensor([row[col] for col in df.columns], dtype=torch.float32)  # Shape: [num_columns, 13]
        # Transpose to shape [13, num_columns]
        input_tensor = input_tensor.T
        # Move tensor to the same device as the model
        input_tensor = input_tensor.to(device)
        # Pass through the model
        with torch.no_grad():  # Disable gradient computation for inference
            output_tensor = model(input_tensor)  # Output shape: [13, output_dim]
        # Convert the model output back to a list
        return output_tensor.cpu().tolist()  # Ensure the output is moved to CPU before converting to list
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Apply the function row-wise to create the new column
    df_copy['ARTC_model'] = df_copy.apply(process_row, axis=1)
    return df_copy


def fill_none_with_ffill_bfill(inputs):
    return pd.DataFrame(inputs).bfill().ffill().values.reshape(-1)


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

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import ast
import re

def omit_top_k_outliers(arr, k):
    # Convert the array to a NumPy array if it's not already
    arr = np.array(arr, dtype=np.float64)
    
    # Filter out None or NaN values
    valid_mask = ~np.isnan(arr)
    valid_values = arr[valid_mask]

    # Find the indices of the top k outliers based on sorted order
    sorted_indices = np.argsort(valid_values)
    outlier_indices = sorted_indices[-k:] if k > 0 else []

    # Create a boolean mask that marks the top k outliers for removal
    outlier_mask = np.ones_like(valid_values, dtype=bool)
    outlier_mask[outlier_indices] = False
    
    # Filter out the outliers while preserving the original order
    filtered_values = valid_values[outlier_mask]

    # Return the filtered array with valid values only
    return filtered_values

def mape_adjusted_array(y_true, y_pred, axis=0, zeros=np.nan):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    assert y_true.shape == y_pred.shape
    error = abs(y_true - y_pred)
    non_zero_mask = (y_true != 0).ravel()
    percentage_error = np.full_like(error, zeros, dtype=np.float32)
    percentage_error.flat[non_zero_mask] = error.flat[non_zero_mask] / abs(y_true.flat[non_zero_mask])
    return percentage_error

def construct_df_future(skus, date_range):
    df_all = []
    for sku in skus:
        df_new = pd.DataFrame({
            'unique_id': sku,
            'ds': date_range,
            'y': [0] * len(date_range)  # Or set 'y' to NaN if future data is not available
        })
        df_all.append(df_new)
    return pd.concat(df_all, ignore_index=True)

def ProcessedData2TS(df_ori):
    # Convert the 'Date' column to datetime format
    df_ori['Date'] = pd.to_datetime(df_ori['Date'], format='%Y-%m-%d')

    # Pivot the table to get the desired format
    df_pivot = df_ori.pivot_table(index='SKU Code', columns=df_ori['Date'].dt.to_period('M'), values='Quantity', fill_value=np.nan)

    # Flatten the column MultiIndex to have it as regular columns
    df_pivot.columns = df_pivot.columns.astype(str)

    return df_pivot

def save_df_if_not_exist(df, file_path_1, overwrite=False):
    if os.path.exists(file_path_1) and not overwrite:
        print(f"The file '{file_path_1}' exists.")
    else:
        # Determine the file extension
        file_extension = os.path.splitext(file_path_1)[1].lower()

        # Save the DataFrame according to the file extension
        if file_extension == '.csv':
            df.to_csv(file_path_1, index=False)
        elif file_extension == '.xlsx':
            df.to_excel(file_path_1, index=False)
        elif file_extension == '.json':
            df.to_json(file_path_1)
        elif file_extension == '.parquet':
            df.to_parquet(file_path_1)
        else:
            raise ValueError(f"Unsupported file extension: '{file_extension}'")

        print(f"The file '{file_path_1}' has been saved.")

def plot_accuracy_adjusted_mape(y_true, y_pred, outlier_fraction=0.05, plotter=True, title_prefix="Averaged accuracy for naive prediction of 13-week data"):
    """
    Function to calculate and plot the accuracy adjusted MAPE with a trend line.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    outlier_fraction (float): Fraction of outliers to omit.

    Returns:
    None
    """
    # Calculate MAPE and adjust it
    mape_all = mape_adjusted_array(y_true=y_true, y_pred=y_pred)
    mape_all_filtered = omit_top_k_outliers(mape_all, int(outlier_fraction * mape_all.shape[0]))
    accuracy_adjusted_mape = 1 / (1 + mape_all_filtered)

    # Calculate the average accuracy
    average_accuracy = np.nanmean(accuracy_adjusted_mape)
    
    if plotter:
        # Plot the data
        plt.figure(figsize=(20, 5))
        plt.plot(accuracy_adjusted_mape, 'o-', label='Accuracy Adjusted MAPE')

        # Plot the horizontal line for the average accuracy
        plt.axhline(y=average_accuracy, color='r', linestyle='--', label=f'Average Accuracy = {average_accuracy*100:.1f}%')

        # Add title and labels
        plt.title(f"{title_prefix} = {average_accuracy*100:.1f}%")
        plt.xlabel('Index')
        plt.ylabel('Accuracy')

        # Display the legend in the bottom right corner
        plt.legend(loc='lower right')

        # Display the plot
        plt.show()
    
    return average_accuracy


def string_to_number_list(string_data):
    # Step 1: Remove the outer square brackets and strip unnecessary characters
    cleaned_string = string_data.strip("[]'")
    
    # Step 2: Use ast.literal_eval to safely parse the string into a Python list
    list_of_numbers = list(ast.literal_eval(cleaned_string))
    
    return list_of_numbers


def string_to_datetime_list(string_datetime):
    # Extract the date strings using a regular expression
    date_strings = re.findall(r'\d{4}-\d{2}-\d{2}', string_datetime)
    date_range = pd.to_datetime(date_strings)
    return date_range

# Define metrics
def mse(y_pred, y_true):
  return np.mean(np.square(y_pred - y_true),  keepdims=True)

def mae(y_pred, y_true):
  return np.mean(np.abs(y_pred - y_true), keepdims=True)


def MAPE(y_true, y_pred): 
    ### omit X/0 mapes
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.nanmean(np.abs((y_true - y_pred) / y_true))

def convert_string_to_nparray(string_array):
    # Convert the string to a valid Python list (if needed)
    if isinstance(string_array, str):
        string_array = ast.literal_eval(string_array)
    
    # Convert the list to a numpy array
    return np.array(string_array, dtype=float)

def process_price_data(file_path, freq='M', index='date', columns='sku_code', values='weighted_avg_unit_price_discount'):
    """
    Function to load, process, and pivot a price DataFrame based on the provided date type ('weekly' or 'monthly').

    Parameters:
    file_path (str): Path to the CSV file.
    date_type (str): Specify whether the data is 'weekly' or 'monthly'. Default is 'weekly'.
    
    Returns:
    pd.DataFrame: Processed DataFrame with appropriate date index.
    """
    # Load the data
    df_price = pd.read_csv(file_path)
    df_temp = df_price.pivot(index=index, columns=columns, values=values)
    # Determine columns based on data type
    if freq == 'W-FRI' or freq == 'W-FRI':
        df_temp.index = pd.to_datetime(df_temp.index).to_period('W-FRI').to_timestamp('W-FRI')
    elif freq == 'M' or freq == 'm':
        df_temp.index = pd.to_datetime(df_temp.index).to_period('M').to_timestamp('M') - pd.offsets.MonthBegin(1)
    
    # Sort the rows based on the index and handle missing values
    df_temp.sort_index(inplace=True)
    df_temp.fillna(method='ffill', axis=0, inplace=True)
    df_temp.fillna(method='bfill', axis=0, inplace=True)

    return df_temp

def process_weather_data(file_path, freq='W-FRI', index='date', columns='facility', values='temperature_average'):
    """
    Function to load, process, and pivot weather data, computing the average temperature across facilities or states.

    Parameters:
    file_path (str): Path to the CSV file.
    date_type (str): Specify whether the data is 'daily' or 'monthly'. Default is 'daily'.
    
    Returns:
    pd.DataFrame: Processed DataFrame with temperature averages.
    """

    # Load the data
    df_weather = pd.read_csv(file_path)
    df = df_weather.pivot(index=index, columns=columns, values=values)
    # Pivot and adjust date index based on the date type
    if freq == 'W' or freq == 'w':
        df.index = pd.to_datetime(df.index)
    elif freq == 'M' or freq == 'm':
        df.index = pd.to_datetime(df.index) - pd.offsets.MonthBegin(1)

    # Compute the average temperature and sort by date
    if values == 'temperature_average':
        df['T_average'] = df.mean(axis=1)
    elif values =='precipitation':
        df['P_average'] = df.mean(axis=1)
    df.sort_index(inplace=True)
    
    return df


def plot_boxplot_with_stats(df, figsize=(8, 6), xlim=(0.5, 4.8), ylim=(0.6, 0.68), rotation=45, fig_title='Box Plot of Columns in DataFrame', baseline=0.6):
    """
    Plot a boxplot for all columns in the provided DataFrame and annotate with min, max, median, and average values.
    Adds a horizontal line at a specified baseline.

    Parameters:
    df (pd.DataFrame): DataFrame to plot the boxplot from.
    figsize (tuple): Size of the figure.
    xlim (tuple): Limits for the x-axis.
    ylim (tuple): Limits for the y-axis.
    rotation (int): Rotation angle for x-axis labels.
    fig_title (str): Title for the plot.
    baseline (float): Value for the horizontal baseline. Default is 0.6.
    """
    # Plot the box plot for all columns in the DataFrame
    plt.figure(figsize=figsize)
    plt.boxplot(df.values, labels=df.columns)
    
    # Annotate with min, max, median, and average values
    for i, column in enumerate(df.columns):
        column_data = df[column]
        plt.text(i + 1, column_data.min() - 0.001, f'Min: {column_data.min()*100:.2f}%', ha='center', va='top', fontsize=8)
        plt.text(i + 1, column_data.max() + 0.001, f'Max: {column_data.max()*100:.2f}%', ha='center', va='bottom', fontsize=8)
        plt.text(i + 1.5, column_data.median(), f'Med: {column_data.median()*100:.2f}%', ha='center', va='center', fontsize=8, color='red')
        plt.text(i + 1, np.min(ylim), f'Avg: {100*column_data.mean():.2f}%', ha='center', va='bottom', fontsize=8, color='blue')

    # Add a horizontal line at the baseline
    plt.axhline(y=baseline, color='green', linestyle='--', linewidth=1, label=f'Baseline: {baseline*100: .2f}%')
    
    # Add labels and title
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title(fig_title)
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Rotate x-axis labels if needed and adjust layout
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.legend()

    # Show the plot
    plt.show()

# Example usage:
# plot_boxplot_with_stats(df_results_all, baseline=0.65)



# Defining the function based on the given equation
def WFA_market_TCCC(y_true, y_pred):
    """
    Calculates the Market Level WFA using the given formula.

    Parameters:
    y (list or array-like): Actual values
    y_hat (list or array-like): Predicted values

    Returns:
    float: Market level WFA value
    """
    
    # Replace negative values with 0
    y_pred[y_pred < 0] = 0
    
    # Calculate the numerator: sum of absolute differences between actual and predicted values
    numerator = sum(abs(y_j - y_hat_j) for y_j, y_hat_j in zip(y_true, y_pred))
    
    # Calculate the denominator: sum of the actual and predicted values, scaled by 1/2
    denominator = (1 / 2) * sum(y_j + y_hat_j for y_j, y_hat_j in zip(y_true, y_pred))
    
    # Calculate the WFA_market using the max condition
    WFA_market_value = max(1 - (numerator / denominator), 0)
    
    return WFA_market_value

# Defining the function based on the given equation
def WFA_market_TCCC(y_true, y_pred): 
    # Replace negative values with 0
    # y_pred[y_pred < 0] = 0
    
    # Calculate the numerator: sum of absolute differences between actual and predicted values
    numerator = sum(abs(y_j - y_hat_j) for y_j, y_hat_j in zip(y_true, y_pred))
    
    # Calculate the denominator: sum of the actual and predicted values, scaled by 1/2
    denominator = sum(abs(y_j) for y_j in y_true)
    
    # Calculate the WFA_market using the max condition
    WFA_market_value = max(1 - (numerator / denominator), 0)
    
    return WFA_market_value

def save_dataframe_to_json(df, file_path):
    """
    Save a DataFrame to a JSON file, ensuring the directory exists.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - file_path (str): The file path to save the JSON file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the DataFrame to JSON
    df.to_json(file_path, orient='records', lines=True, date_format='iso', double_precision=15)
    print(f"DataFrame successfully saved to {file_path}")


def prepare_training_data_multivar(
    df_train_sales, 
    df_temperature, 
    df_price, 
    training_date_range
):
    results = []
    
    # Iterate over each row in the sales DataFrame
    for idx, row in df_train_sales.iterrows():
        sku_facility = row.name
        sales_data = row.values.tolist()
        num_sales = len(sales_data)
        
        # Match the length of temperature and price data to sales data
        temperature_data = df_temperature[sku_facility].values.tolist()[:num_sales]
        price_data = df_price[sku_facility].values.tolist()[:num_sales]
        dates = training_date_range

        # Create a temporary DataFrame for the current SKU-facility
        temp_df = pd.DataFrame({
            'unique_id': sku_facility,
            'ds': dates,
            'y': sales_data,
            'temperature_ex': temperature_data,
            'promo_ex': price_data
        })

        results.append(temp_df)
    
    # Concatenate all temporary DataFrames into one final DataFrame
    final_df = pd.concat(results, ignore_index=True)
    
    return final_df

def prepare_training_data_univar(
    df_train_sales, 
    training_date_range
):
    results = []
    
    # Iterate over each row in the sales DataFrame
    for idx, row in df_train_sales.iterrows():
        sku_facility = row.name
        sales_data = row.values.tolist()
        
        # Match the length of temperature and price data to sales data
        dates = training_date_range

        # Create a temporary DataFrame for the current SKU-facility
        temp_df = pd.DataFrame({
            'unique_id': sku_facility,
            'ds': dates,
            'y': sales_data
        })

        results.append(temp_df)
    
    # Concatenate all temporary DataFrames into one final DataFrame
    final_df = pd.concat(results, ignore_index=True)
    
    return final_df

def prepare_training_data(
    df_train_sales, 
    training_date_range
):
    results = []
    
    # Iterate over each row in the sales DataFrame
    for idx, row in df_train_sales.iterrows():
        sku_facility = row.name
        sales_data = row.values.tolist()
        
        # Match the length of temperature and price data to sales data
        dates = training_date_range

        # Create a temporary DataFrame for the current SKU-facility
        temp_df = pd.DataFrame({
            'unique_id': sku_facility,
            'ds': dates,
            'y': sales_data
        })

        results.append(temp_df)
    
    # Concatenate all temporary DataFrames into one final DataFrame
    final_df = pd.concat(results, ignore_index=True)
    
    return final_df

def prepare_weekly_inputs_multivar(
    df_sales,
    df_temperature,
    df_price,
    testing_context_start_date_ori,
    horizon,
    LAG,
    context_size,
    testing_len = None
):
    # Initialize lists to store results
    if testing_len is None:
        testing_len = horizon
        
    sku_fac_list, data_range_list, data_range_ex_list = [], [], []
    temperature_ex_list, price_ex_list, context_list, sales_N_list = [], [], [], []

    # Iterate over each SKU-facility in the sales DataFrame
    for idx, row in df_sales.iterrows():
        sku_facility = row.name
        
        for rolling_id in range(testing_len - LAG + 1):
            sku_fac_list.append(sku_facility)
            
            # Calculate the start date for the rolling window
            start = pd.to_datetime(testing_context_start_date_ori) + pd.offsets.MonthEnd(rolling_id)
            
            # Historical context date range
            date_range = pd.date_range(start=start, periods=context_size, freq='M')
            data_range_list.append(date_range)
            
            # Sales data for the context
            w1, w2 = date_range[0], date_range[-1]
            sales_data_context = row.loc[w1:w2].to_list()
            context_list.append(sales_data_context)
            
            # Future sales data (sales_N)
            w1, w2 = pd.to_datetime(date_range[-1]) + pd.offsets.MonthEnd(1), pd.to_datetime(date_range[-1]) + pd.offsets.MonthEnd(3)
            sales_N = row.loc[w1:w2].to_list()
            sales_N_list.append(sales_N)
            
            # Extended date range for multivariate inputs
            date_range_mv = pd.date_range(start=start, periods=context_size + horizon, freq='M')
            data_range_ex_list.append(date_range_mv)
            
            # Temperature data for the extended range
            filtered_df_temperature = select_rows_by_date_range(
                df_temperature[sku_facility], 
                start_date=date_range_mv[0], 
                end_date=date_range_mv[-1]
            )
            temperature_list = filtered_df_temperature.to_list()
            temperature_ex_list.append(temperature_list)
            
            # Price data for the extended range
            filtered_df_price = select_rows_by_date_range(
                df_price[sku_facility], 
                start_date=date_range_mv[0], 
                end_date=date_range_mv[-1]
            )
            price_list = filtered_df_price.to_list()
            price_ex_list.append(price_list)
    
    # Combine all the collected data into a single DataFrame
    df_all_inputs_weekly = pd.DataFrame({
        "unique_id": sku_fac_list,
        "data_range": data_range_list,
        "data_range_ex": data_range_ex_list,
        "temperature_ex": temperature_ex_list,
        "promo_ex": price_ex_list,
        "context": context_list,
        "sales_N": sales_N_list
    })
    
    return df_all_inputs_weekly



def extend_weather_data_for_facilities(weather_data, date_st, date_end):
    """
    Extend the weather dataset for multiple facilities to cover a specific date range.
    If a date is missing for a facility, use the data for the same day in another year.

    Parameters:
    - weather_data (pd.DataFrame): Original weather dataset with columns ['facility', 'date', ...].
    - date_st (str or pd.Timestamp): Start date of the desired range (inclusive).
    - date_end (str or pd.Timestamp): End date of the desired range (inclusive).

    Returns:
    - pd.DataFrame: Extended weather dataset with all dates covered for each facility.
    """
    # Ensure 'date' column is datetime
    weather_data['date'] = pd.to_datetime(weather_data['date'])

    # Generate the full date range
    full_date_range = pd.date_range(start=date_st, end=date_end)

    # Add 'month' and 'day' columns for matching missing dates
    weather_data['month'] = weather_data['date'].dt.month
    weather_data['day'] = weather_data['date'].dt.day

    # Create an empty DataFrame to store the extended data
    extended_weather_data = pd.DataFrame()

    # Process each facility
    for facility in weather_data['facility'].unique():
        # Filter data for the facility
        facility_data = weather_data[weather_data['facility'] == facility]

        # Create a DataFrame for the full date range for the facility
        facility_extended = pd.DataFrame({'date': full_date_range})
        facility_extended['month'] = facility_extended['date'].dt.month
        facility_extended['day'] = facility_extended['date'].dt.day

        # Merge on 'month' and 'day' to fill missing dates
        facility_extended = facility_extended.merge(
            facility_data,
            on=['month', 'day'],
            how='left',
            suffixes=('', '_original')
        )

        # Fill missing values using the closest available data
        facility_extended = facility_extended.ffill().bfill()

        # Add the facility column
        facility_extended['facility'] = facility

        # Drop temporary columns
        facility_extended = facility_extended.drop(columns=['month', 'day'])

        # Append to the main DataFrame
        extended_weather_data = pd.concat([extended_weather_data, facility_extended], ignore_index=True)

    return extended_weather_data



def calculate_mean_variability_sku_facility(df_all):
    # Group by 'SKU_facility' and calculate mean and variability
    grouped = df_all.groupby('SKU_facility')['acc'].agg(
        mean_acc='mean',
        variability_acc='std'  # Standard deviation as variability measure
    ).reset_index()
    ## hyQ should it be only mean to conpute the group accuracy.
    # Fill NaN values in variability (e.g., for single data points)
    grouped['variability_acc'] = grouped['variability_acc'].fillna(0)

    return grouped

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
    mask_zero_true = (y_true == 0) & (y_pred < 0.5)
    acc[mask_zero_true] = 1

    mask_zero_true = (y_true == 0) & (y_pred >= 0.5)
    acc[mask_zero_true] = 0

    # Handle normal cases where y_true != 0
    mask_non_zero = y_true != 0
    acc[mask_non_zero] = 1 - abs(y_pred[mask_non_zero] -
                                y_true[mask_non_zero]) / abs(y_true[mask_non_zero])
    acc = np.maximum(acc, 0)

    # Convert to list and return
    return acc.tolist()

def compute_model_accuracy(df, true_col, start_col_idx, lag, metric_func):

    # Extract the true target values
    y_true = np.array(df[true_col].to_list())[:, lag-1]

    # Extract model names
    model_names = df.columns[start_col_idx:]

    # Initialize lists to store results
    X_all, results = [], []
    y_pred_all = []
    # Loop through each model to compute metrics
    for model in model_names:
        # Extract predictions at the specified lag
        y_pred = np.array([arr[lag - 1] for arr in df[model].values])
        y_pred_all.append(y_pred)

        # Compute the accuracy metric
        if y_pred[0] is not None:
            metric_result = metric_func(y_true=y_true, y_pred=y_pred)
            results.append(metric_result)
        else:
            results.append(0)

    X_all = np.hstack(y_pred_all) # all model prediction
    Y_all = np.array([row for row in df[true_col].values])[:, (lag-1):lag]
    # Create a DataFrame for accuracy results
    df_accuracy = pd.DataFrame(results, index=model_names, columns=['WFA'])
    return df_accuracy, X_all, Y_all

# Define the objective function
def objective_function(W, X, Y):
    # Reshape Y to ensure correct broadcasting
    Y = Y.reshape(-1, 1)  # Convert to (100, 1)
    losses = np.sum(np.abs(np.dot(X, W.T) - Y), axis=0)  # MAE for all particles
    return losses


# Initialize particles
def initialize_particles(n_particles, dimensions, lower_bound, upper_bound):
    positions = np.random.uniform(lower_bound, upper_bound, (n_particles, dimensions))
    velocities = np.random.uniform(-abs(upper_bound - lower_bound), abs(upper_bound - lower_bound), (n_particles, dimensions))
    return positions, velocities

# Update particle velocities and positions
def update_particles(positions, velocities, personal_best_positions, global_best_position, inertia, cognitive, social):
    r1 = np.random.rand(*positions.shape)  # Random coefficients for cognitive component
    r2 = np.random.rand(*positions.shape)  # Random coefficients for social component

    cognitive_component = cognitive * r1 * (personal_best_positions - positions)
    social_component = social * r2 * (global_best_position - positions)
    velocities = inertia * velocities + cognitive_component + social_component
    positions = positions + velocities
    return positions, velocities

# PSO Algorithm
def pso_with_early_stopping(
    X,
    Y,
    n_particles=30,
    dimensions=10,
    lower_bound=-1,
    upper_bound=1,
    max_iters=100,
    inertia=0.7,
    cognitive=1.5,
    social=1.5,
    patience=10,
    tol=1e-6
):
    # Initialize particles
    positions, velocities = initialize_particles(n_particles, dimensions, lower_bound, upper_bound)
    personal_best_positions = positions.copy()
    personal_best_scores = objective_function(personal_best_positions, X, Y)
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    # Track fitness evolution and early stopping
    gbest_history = []
    best_score_improvement_count = 0  # To track improvement across generations

    for iteration in range(max_iters):
        # Evaluate fitness
        fitness = objective_function(positions, X, Y)

        # Update personal bests
        for i in range(n_particles):
            if fitness[i] < personal_best_scores[i]:
                personal_best_scores[i] = fitness[i]
                personal_best_positions[i] = positions[i]

        # Update global best
        min_index = np.argmin(personal_best_scores)
        if personal_best_scores[min_index] < global_best_score - tol:
            global_best_score = personal_best_scores[min_index]
            global_best_position = personal_best_positions[min_index]
            best_score_improvement_count = 0  # Reset improvement count
        else:
            best_score_improvement_count += 1  # Increment if no improvement

        # Early stopping condition
        if best_score_improvement_count >= patience:
            print(f"Early stopping triggered at iteration {iteration + 1}.")
            break

        # Update velocities and positions
        positions, velocities = update_particles(
            positions, velocities, personal_best_positions, global_best_position, inertia, cognitive, social
        )

        # Track the global best score
        gbest_history.append(global_best_score)

        # Print progress
        # print(f"Iteration {iteration + 1}/{max_iters}, Best Loss: {global_best_score:.6f}")

    return global_best_position, global_best_score, gbest_history


import numpy as np

def pso_with_early_stopping(
    X,
    Y,
    n_particles=30,
    dimensions=10,
    lower_bound=-1,
    upper_bound=1,
    max_iters=100,
    inertia=0.7,
    cognitive=1.5,
    social=1.5,
    patience=10,
    tol=1e-6
):
    # Initialize particles
    positions, velocities = initialize_particles(n_particles, dimensions, lower_bound, upper_bound)
    personal_best_positions = positions.copy()
    personal_best_scores = objective_function(personal_best_positions, X, Y)
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    # Track fitness evolution and early stopping
    gbest_history = []
    best_score_improvement_count = 0  # To track improvement across generations

    for iteration in range(max_iters):
        # Evaluate fitness
        fitness = objective_function(positions, X, Y)

        # Update personal bests
        for i in range(n_particles):
            if fitness[i] < personal_best_scores[i]:
                personal_best_scores[i] = fitness[i]
                personal_best_positions[i] = positions[i]

        # Update global best
        min_index = np.argmin(personal_best_scores)
        if personal_best_scores[min_index] < global_best_score - tol:
            global_best_score = personal_best_scores[min_index]
            global_best_position = personal_best_positions[min_index]
            best_score_improvement_count = 0  # Reset improvement count
        else:
            best_score_improvement_count += 1  # Increment if no improvement

        # Early stopping condition
        if best_score_improvement_count >= patience:
            print(f"Early stopping triggered at iteration {iteration + 1}.")
            break

        # Update velocities and positions
        positions, velocities = update_particles(
            positions, velocities, personal_best_positions, global_best_position, inertia, cognitive, social
        )

        # Enforce bounds
        positions = np.clip(positions, lower_bound, upper_bound)

        # Track the global best score
        gbest_history.append(global_best_score)

    return global_best_position, global_best_score, gbest_history

def calculate_acc_PSO(W, Y, X):
    # Calculate predictions using the current weight vector
    W = W.reshape(-1, 1)
    predictions = np.dot(X, W).reshape(-1)
    Y = Y.reshape(-1)
    acc_temp = 1 - abs(predictions - Y).sum() / abs(Y).sum()
    acc = max(acc_temp, 0)
    return predictions, acc

def calculate_pred_PSO(W, X):
    # Calculate predictions using the current weight vector
    W = W.reshape(-1, 1)
    predictions = np.dot(X, W).reshape(-1)
    return predictions

def show_iterative_focus_group_results(num_models, lb, ub, sku_facility_domain, horizon, data_path, sku_cat_path, context_size=64, threshold=0.7):
    identifier_temp = f'{data_path}/{sku_facility_domain}_context_size_{context_size}_horizon_{horizon}_num_models_{num_models}_lb_{lb}_ub_{ub}_threshold_{threshold}_model_id_'
    identifier_ori =  f'{data_path}/{sku_facility_domain}_context_size_{context_size}_horizon_{horizon}_num_models_{num_models}_lb_{lb}_ub_{ub}_threshold_{threshold}_model_id_{1}'
    file_path_artc = f'{identifier_ori}/df_ARTC.json'
    file_path_s5 = f'{identifier_ori}/df_all_mixing_training_s5.json'
    print(file_path_artc)
    
    # Read original ARTC file
    df_artc_original = pd.read_json(file_path_artc, orient='records', lines=True)
    df_s5_ori = pd.read_json(file_path_s5, orient='records', lines=True)
    
    # Initialize list to keep track of focused groups
    df_focused_group_all = []
    Num_focused_group_over_iterations = []
    
    # Loop through model IDs
    for model_id in range(1, 20):
        identifier = f'{identifier_temp}{model_id}'
        file_path_focused_group = f'{identifier}/df_focused_group.json'
        
        # Read focused group file
        df_focused_group = pd.read_json(file_path_focused_group, orient='records', lines=True)
        df_focused_group['Iteration'] = model_id
        num_new_focused_SKUs = df_focused_group.shape[0]
        print(f"Number of focused SKUs: {num_new_focused_SKUs}")
        Num_focused_group_over_iterations.append(num_new_focused_SKUs)
        
        # Break if no new focused SKUs
        if num_new_focused_SKUs == 0:
            print("df_focused_group is empty. Breaking the loop.")
            break
        
        df_focused_group_all.append(df_focused_group)

    # Process SKU category data
    if 'Enrich' in sku_cat_path:
        df_sku_cat = pd.read_csv(sku_cat_path)[['ID','ABC_Category_by_qty']]
        df_sku_cat.rename(columns={'ABC_Category_by_qty': 'ABC_Category'}, inplace=True)
    else:
        df_sku_cat = pd.read_csv(sku_cat_path)[['ID','ABC_Category']]

    # Perform calculations on original ARTC dataframe
    df_results = calculate_mean_variability_sku_facility(df_artc_original)
    
    # Map categories
    df_results['Category'] = df_results['SKU_facility'].map(
        df_sku_cat.set_index('ID')['ABC_Category']
    )
    
    # Check for missing SKUs and drop them
    nan_rows = df_results[df_results.isnull().any(axis=1)]
    print(f"There are total {nan_rows.shape[0]} missing SKUS")
    df_results = df_results.dropna()

    # Map category to colors
    color_mapping = {'A': 'red', 'B': 'blue', 'C': 'cyan'}
    df_results['color'] = df_results['Category'].replace(color_mapping)

    # Combine all focused groups
    df_focused_group_final = pd.concat(df_focused_group_all, axis=0)
    print(f"df_focused_group_final shape is {df_focused_group_final.shape}")
    
    # Print group coverage by category
    print_group_coverage(df_focused_group_final, df_results, Cat='A')
    print_group_coverage(df_focused_group_final, df_results, Cat='B')
    print_group_coverage(df_focused_group_final, df_results, Cat='C')
    return df_focused_group_final, df_results, df_artc_original, df_s5_ori, Num_focused_group_over_iterations

def print_group_coverage(df_focused_group, df_results_combined, Cat ='A'):
    A_focused = df_focused_group[df_focused_group['Category'] == Cat] 
    A_all = df_results_combined[df_results_combined['Category'] == Cat]
    # Calculate coverage as the ratio of the focused group to the combined DataFrame
    coverage_percentage = (A_focused.shape[0] / A_all.shape[0]) * 100
    print(f"Current coverage Cat-{Cat} is {coverage_percentage: .2f}%")
    