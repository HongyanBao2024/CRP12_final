import pandas as pd
import numpy as np


def _print_versions():
    '''Prints the versions of pandas, numpy, and seaborn'''
    print("pandas version:", pd.__version__)
    print("numpy version:", np.__version__)
    #print("seaborn version:", sns.__version__)


def _load_data_demand(demand_data,selected_column,unique_num_limit):
    '''Loads and prepares the data'''
    demand_data.columns = selected_column
    demand_data['ds'] = pd.to_datetime(demand_data['ds'], dayfirst=True)
    unique_id_counts = demand_data['unique_id'].value_counts()
    valid_unique_ids = unique_id_counts[unique_id_counts > unique_num_limit].index # give the number limit 100 for each SKU
    demand_data = demand_data[demand_data['unique_id'].isin(valid_unique_ids)]
    return demand_data


def _add_weather_data(demand_data, weather_data_path):
    '''Integrates weather data with the main demand dataset'''
    demand_data['facility'] = demand_data['unique_id'].str.split('-').str[1]
    demand_data['ds'] = pd.to_datetime(demand_data['ds'])
    weather_data = pd.read_csv(weather_data_path)
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    combined_data = pd.merge(demand_data, weather_data, left_on=['facility', 'ds'], right_on=['facility', 'date'], how='left')
    return combined_data


def _select_variate(demand_sku_weekly, weather=True, other_f=True):
    """
    Selects specified features for the dataset.

    Args:
    - demand_sku_weekly (pd.DataFrame): DataFrame containing the SKU weekly demand data.
    - weather (bool): Include weather features if True.
    - other_f (bool): Include other features (discount, order count) if True.

    Returns:
    - X_ts (pd.DataFrame): DataFrame with selected features.
    - Y_ts (pd.DataFrame): DataFrame with target variable 'y'.
    """
    feature_columns = ['unique_id', 'ds']

    if weather:
        feature_columns += ['temperature_max']

    if other_f:
        feature_columns += ['discount_percentage', 'unit_price_discount', 'order_count']

    X_ts = demand_sku_weekly[feature_columns]
    Y_ts = demand_sku_weekly[['unique_id', 'ds', 'y']]

    # Ensure correct data types
    X_ts['ds'] = pd.to_datetime(X_ts['ds'])
    Y_ts['ds'] = pd.to_datetime(Y_ts['ds'])
    X_ts['unique_id'] = X_ts['unique_id'].astype(str)

    return X_ts, Y_ts


def create_roll_data(df_kbl, lag=2,weather=False,other_f=False):
    start_size = lag + 12
    end_size = lag
    X_ts, Y_ts = _select_variate(df_kbl, weather, other_f)
    versions = {}

    for test_size in range(start_size, end_size - 1, -1):
        X_train_list, X_test_list = [], []
        Y_train_list, Y_test_list = [], []

        for uid, group in X_ts.groupby('unique_id'):
            group = group.sort_values('ds')
            X_test_list.append(group.iloc[-test_size:])
            X_train_list.append(group.iloc[:-test_size])

        for uid, group in Y_ts.groupby('unique_id'):
            group = group.sort_values('ds')
            Y_test_list.append(group.iloc[-test_size:])
            Y_train_list.append(group.iloc[:-test_size])

        X_train = pd.concat(X_train_list).reset_index(drop=True)
        X_test = pd.concat(X_test_list).reset_index(drop=True)
        Y_train = pd.concat(Y_train_list).reset_index(drop=True)
        Y_test = pd.concat(Y_test_list).reset_index(drop=True)

        train = Y_train.merge(X_ts, how='left', on=['unique_id', 'ds'])

        versions[test_size] = {
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test,
            'train': train
        }

        #print(f"Test size: {test_size}")
        # print("X_train shape:", X_train.shape)
        # print("X_test shape:", X_test.shape)
        # print("Y_train shape:", Y_train.shape)
        # print("Y_test shape:", Y_test.shape)

    return versions


def mape_adjusted(y, y_pred, axis=0, zeros=np.nan,evaluate_number=13):
    """
    Mean Absolute Percentage Error adjusted for actual zero values

    Parameters
    ----------
    y : array_like, includes lists, lists of tuples, tuples,
    tuples of tuples, tuples of lists and ndarrays.
      The actual value.
    y_pred : array_like
       The predicted value.
    axis : int
       Axis along which the summary statistic is calculated
    zeros : float
       Value to assign to error where y is zero

    Returns
    -------
    mape : ndarray or float
       Mean Absolute Percentage Error along given axis.
    """
    y_pred = np.asarray(y_pred)
    y = np.asarray(y)
    if np.all(y == 0):
        return np.nan
    assert y.shape == y_pred.shape
    error = abs(y - y_pred)
    non_zero_mask = (y != 0).ravel()
    percentage_error = np.full_like(error, zeros, dtype=np.float32)
    percentage_error.flat[non_zero_mask] = error.flat[non_zero_mask] / abs(y.flat[non_zero_mask])
    mape_adjusted_for_zero = np.nanmean(percentage_error, axis=axis) * 100
    return mape_adjusted_for_zero.round(6)


def accuracy_from_mape_adjusted(mape_adjusted_value):
    """
    Accuracy calculated from Mean Absolute Percentage Error adjusted to a range of [0, 100].
    Formula: accuracy = max(0, 100 - mape_adjusted_value)
    """
    if np.isnan(mape_adjusted_value):
        return np.nan
    return max(0, 100 - mape_adjusted_value)


def extract_second_point_from_forecasts(all_results_versions):
    all_second_points = []

    for test_size, results_data in all_results_versions.items():
        results = results_data

        # For each unique_id, extract the second forecasted point
        for uid in results['SKU'].unique():
            # Filter data for this unique_id
            uid_results = results[results['SKU'] == uid]

            # Ensure that there are at least 2 points, then extract the second one
            if len(uid_results) >= 2:
                second_point = uid_results.iloc[1]  # Extract the second point (index 1)
                second_point['Test_Size'] = test_size  # Add the test_size for reference
                all_second_points.append(second_point)

    # Concatenate all second points into a single DataFrame
    second_points_df = pd.DataFrame(all_second_points).reset_index(drop=True)

    return second_points_df

def eval_best_model(prediction_df_roll):
    """
    Calculate the average MAPE and accuracy for each SKU and model combination
    over the last 13 points in the input DataFrame.

    Args:
        prediction_df_roll (pd.DataFrame): DataFrame containing predictions, dates, actuals,
                                           and grouped by SKU and model.

    Returns:
        pd.DataFrame: A DataFrame containing the SKU, model, average MAPE, and accuracy.
    """
    # Apply transformations to extract specific elements
    prediction_df_roll['Predictions'] = prediction_df_roll['Predictions'].apply(lambda x: x[1])
    prediction_df_roll['Dates'] = prediction_df_roll['Dates'].apply(lambda x: x[1])
    prediction_df_roll['Actuals'] = prediction_df_roll['Actuals'].apply(lambda x: x[1])

    results = []
    grouped = prediction_df_roll.groupby(['SKU', 'Model'])

    for (sku, model), group in grouped:
        # Calculate pointwise MAPE and ACC
        pointwise_mape = abs((group['Actuals'] - group['Predictions']) / group['Actuals']) * 100
        pointwise_acc = (100 - pointwise_mape).clip(lower=0)

        # Filter out invalid values
        valid_indices = group['Actuals'] != 0
        pointwise_mape = pointwise_mape[valid_indices]
        pointwise_acc = pointwise_acc[valid_indices]

        # Calculate average metrics
        avg_mape = pointwise_mape.mean() if not pointwise_mape.empty else None
        avg_acc = pointwise_acc.mean() if not pointwise_acc.empty else None

        # Append results only if MAPE is valid
        if avg_mape is not None and avg_mape != 0:
            results.append({
                'SKU': sku,
                'Model': model,
                'Average_MAPE': avg_mape,
                'Average_ACC': avg_acc
            })

    result_df = pd.DataFrame(results)

    # Select the best model for each SKU based on ACC
    best_models_df = result_df.loc[result_df.groupby('SKU')['Average_ACC'].idxmax()]

    return best_models_df
