import pandas as pd
import numpy as np
from calendar import monthrange
from datetime import datetime, timedelta
import re

def sales_level_transfer(df,level1 = 'Kit #', level2 = 'KC'):
    # Create the level2 column
    df[level2] = df[level1].astype(str) + "_" + df["Country"]

    # Drop original level2 and 'Country' columns
    df.drop(columns=[level1, "Country"], inplace=True)

    # Move 'KC' column to first position if there are more than 10 columns

    df = df[[level2] + [col for col in df.columns if col != level2]]

    # give the output of the unique 'KC'
    df = df.groupby(level2, as_index=False).sum(numeric_only=True)

    return df

def sum_with_nan_transform(group):
    result = group.sum()
    # If the group contains NaN, return NaN for that column
    return result if not result.isna().any() else np.nan


def forecast_level_transfer(df,level1 = 'Kit #', level2 = 'KC'):
    # Create the level2 column
    df[level2] = df[level1].astype(str) + "_" + df["Country"]
    desired_columns = [
    'KC', 'Version', 'Revised Inventory Policy', '2024/09/30', '2024/10/31', '2024/11/30', 
    '2024/12/31', '2025/01/31', '2025/02/28', '2025/03/31', '2025/04/30',
    '2025/05/31', '2025/06/30', '2025/07/31', '2025/08/31', '2025/09/30',
    '2025/10/31', '2025/11/30', '2025/12/31']

    # Reorder columns and drop the rest
    df = df[desired_columns]
    print(df.duplicated(subset=['KC', 'Version','Revised Inventory Policy']).sum())
    # give the output of the unique 'KC'
    # Apply groupby and use sum with NaNs retained
    result = df.groupby(['KC', 'Version', 'Revised Inventory Policy'], as_index=False).agg(lambda x: x.sum() if x.notna().any() else np.nan)

    return result

def overlap_df_basecolumn(df1,df2,cloume_name):

    # Selecting only overlapping "Kit #" values
    overlapping_Kits = set(df1[cloume_name]) & set(df2[cloume_name])
    print('the number of the sales kit', len(list(df1[cloume_name].unique())))
    print('the number of the forecast kit', len(list(df2[cloume_name].unique())))
    print('the number of the overalpping kit', len(list(overlapping_Kits)))

    # Filtering both DataFrames to keep only the overlapping Kit #
    df1_filtered = df1[df1[cloume_name].isin(overlapping_Kits)]
    df2_filtered = df2[df2[cloume_name].isin(overlapping_Kits)]

    return df1_filtered, df2_filtered

def select_allzero(df1,df2,cloume_name):
    # Step 1: Identify columns to check (exclude 'Kit #')
    cols_to_check = df1.columns.difference([cloume_name])

    # Step 2: For each group, keep rows where all values are 0
    valid_kits = (
        df1.groupby(cloume_name)[cols_to_check]
        .apply(lambda x: ((x == 0) | (x.isna())).any().all())
    )

    # Step 3: Filter the original DataFrame to include only those rows
    result_df1 = df1[df1[cloume_name].isin(valid_kits[valid_kits].index)]
    result_df2 = df2[df2[cloume_name].isin(valid_kits[valid_kits].index)]
    

    return result_df1,result_df2

def select_allzero_onefile(df1,cloume_name):
    # Step 1: Identify columns to check (exclude 'Kit #')
    cols_to_check = df1.columns.difference([cloume_name])

    # Step 2: For each group, keep rows where all values are 0
    valid_kits = (
        df1.groupby(cloume_name)[cols_to_check]
        .apply(lambda x: ((x == 0) | (x.isna())).any().all())
    )

    # Step 3: Filter the original DataFrame to include only those rows
    result_df1 = df1[df1[cloume_name].isin(valid_kits[valid_kits].index)]

    return result_df1

def select_nonzero(df1,df2,cloume_name):
    # Step 1: Identify columns to check (exclude 'Kit #')
    cols_to_check = df1.columns.difference([cloume_name])

    # Step 2: For each group, keep rows where not all values in other columns are 0 or NaN
    valid_kits = (
        df1.groupby(cloume_name)[cols_to_check]
        .apply(lambda x: ~((x == 0) | (x.isna())).any().all())
    )

    # Step 3: Filter the original DataFrame to include only those rows
    result_df1 = df1[df1[cloume_name].isin(valid_kits[valid_kits].index)]
    result_df2 = df2[df2[cloume_name].isin(valid_kits[valid_kits].index)]

    return result_df1,result_df2

def select_nonzero_onefile(df1,cloume_name):
    # Step 1: Identify columns to check (exclude 'Kit #')
    cols_to_check = df1.columns.difference([cloume_name])

    # Step 2: For each group, keep rows where not all values in other columns are 0 or NaN
    valid_kits = (
        df1.groupby(cloume_name)[cols_to_check]
        .apply(lambda x: ~((x == 0) | (x.isna())).any().all())
    )

    # Step 3: Filter the original DataFrame to include only those rows
    result_df1 = df1[df1[cloume_name].isin(valid_kits[valid_kits].index)]

    return result_df1

def fill_version(df):
    # Define all possible versions
    all_versions = ['D09 2024', 'D10 2024', 'D11 2024', 'D12 2024', 'D01 2025']

    # Get all unique Kit # values
    kit_numbers = df["Kit #"].unique()

    # Create a full index with all Kit # and Version combinations
    full_index = pd.MultiIndex.from_product([kit_numbers, all_versions], names=["Kit #", "Version"])

    # Reindex DataFrame to ensure all combinations exist
    df = df.set_index(["Kit #", "Version"]).reindex(full_index).reset_index()

    # Define fill logic dynamically for 16 columns
    num_columns = 16  # Update this if you need more columns

    fill_logic = {
        "D09 2024": [0] + [0] * (num_columns - 1),
        "D10 2024": [None] + [0] * (num_columns - 1),
        "D11 2024": [None, None] + [0] * (num_columns - 2),
        "D12 2024": [None, None, None] + [0] * (num_columns - 3),
        "D01 2025": [None, None, None, None] + [0] * (num_columns - 4),
    }

    # Apply fill logic based on Version
    for version, fill_values in fill_logic.items():
        df.loc[df["Version"] == version, df.columns[2:]] = df.loc[df["Version"] == version, df.columns[2:]].fillna(
            pd.Series(fill_values, index=df.columns[2:])
        )

    return df

def fill_inventorypolicy(df1,df2):
    # Mapping 'Inventory Policy1' based on 'Kit #' from df2
    policy_map = df2.set_index("Kit #")["Inventory Policy1"].to_dict()

    # Adding 'Inventory Policy1' column to df1 while keeping the row count the same
    df1["Inventory Policy1"] = df1["Kit #"].map(policy_map)

    # Reordering columns to place 'Inventory Policy1' after 'Version'
    cols = ["Kit #", "Version", "Inventory Policy1"] + [col for col in df1.columns if col not in ["Kit #", "Version", "Inventory Policy1"]]
    df1 = df1[cols]

    return df1


def widetolong(df_sales_ori_wide_nonzero, level):
    # from wide type of data to long type of data
    df_sales_ori_long_nonzero = df_sales_ori_wide_nonzero.melt(id_vars=[level], var_name="Month_Year", value_name="Sales")

    # Ensuring 'Month_Year' maintains the same order as the original column order
    df_sales_ori_wide_nonzero[level] = df_sales_ori_wide_nonzero[level].astype(str)  # Ensure consistent data types for sorting
    df_sales_ori_long_nonzero["Month_Year"] = pd.Categorical(df_sales_ori_long_nonzero["Month_Year"], categories=list(df_sales_ori_wide_nonzero.keys())[1:], ordered=True)

    # Sorting to maintain the correct column order for each Product ID
    df_sales_ori_long_nonzero = df_sales_ori_long_nonzero.sort_values(by=[level, "Month_Year"]).reset_index(drop=True)


    return df_sales_ori_long_nonzero

# Function to convert "FM-MM-YYYY" to "YYYY/MM/the last day of the month"
def convert_column_name(column_name):
    if not column_name.startswith("FM-"):
        return column_name  # Keep "Product ID" unchanged
    
    parts = column_name.split('-')
    if len(parts) != 3:
        return column_name  # Return as is if format is unexpected

    try:
        month = int(parts[1])  # Extract month as integer
        year = int(parts[2])  # Extract year as integer

        # Validate month range
        if month < 1 or month > 12:
            return column_name  # Skip invalid months

        # Get the last day of the given month
        last_day = monthrange(year, month)[1]

        return f"{year}/{month:02d}/{last_day}"
    
    except ValueError:
        return column_name 
    
def fw_to_friday(fw_label):
    match = re.match(r'^FW-(\d{2})-(\d{4})$', fw_label)
    if match:
        week, year = int(match.group(1)), int(match.group(2))
        try:
            monday = datetime.fromisocalendar(year, week, 1)
            friday = monday + timedelta(days=4)
            return friday.strftime('%d/%m/%Y')
        except ValueError:
            # Handles invalid week numbers (e.g., week 53 in a year without 53 weeks)
            return fw_label
    return fw_label

def fw_to_sunday_date(fw_label):
    match = re.match(r'^FW-(\d{2})-(\d{4})$', fw_label)
    if match:
        week, year = int(match.group(1)), int(match.group(2))
        try:
            monday = datetime.fromisocalendar(year, week, 1)
            sunday = monday + timedelta(days=6)
            return sunday.strftime('%d/%m/%Y')
        except ValueError:
            # Handle invalid week numbers (e.g., FW-53 in years with only 52 ISO weeks)
            for day in range(28, 32):  # December 28–31
                try:
                    date = datetime(year, 12, day)
                    if date.isocalendar()[1] == week:
                        sunday = date + timedelta(days=(6 - date.weekday()))
                        return sunday.strftime('%d/%m/%Y')
                except:
                    continue
            return fw_label
    return fw_label


