import yaml
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

def read_data_from_csv(file_path):
    """
    Reads data from a CSV file and returns it as a pandas DataFrame.
    
    Args:
        file_path (str): The path to the CSV file to be read.
    
    Returns:
        pd.DataFrame: A pandas DataFrame containing the CSV data, or None if an error occurred.
    """
    logging.info("Fetching data from CSV file")
    
    # Check if the file exists before attempting to load
    logging.info("Checking if the file exists")
    
    try:
        # Attempt to load the CSV file using pandas
        data = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {len(data)} rows and {len(data.columns)} columns from {file_path}")
        
        # Check if the loaded data has zero rows
        if data.shape[0] == 0:
            logging.error(f"File {file_path} has zero rows.")
            return None
        
        return data
    
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
    except pd.errors.ParserError:
        logging.error(f"Error parsing CSV file {file_path}. The file might be malformed.")
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading the file: {e}")
    
    return None


def load_yaml_config(filename):
    """
    Loads configuration data from a YAML file.
    
    Args:
        filename (str): The path to the YAML file to be loaded.
    
    Returns:
        dict: The data from the YAML file loaded into a dictionary, or None if an error occurred.
    """
    try:
        # Open the YAML file and load it into a dictionary
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)  # Load YAML content safely
        logging.info(f"Successfully loaded data from {filename}")
        return data
    
    except FileNotFoundError:
        logging.error(f"File {filename} not found.")
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {filename}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    
    return None


def get_new_version_from_existing_forecasts(forecast_output_table_path, version_prefix):
    """
    Determines the new version of the forecast by extracting the highest existing version from a forecast CSV.
    
    Args:
        forecast_output_table_path (str): The path to the CSV file containing the existing forecasts.
        version_prefix (str): The prefix to use when determining the forecast version.
    
    Returns:
        tuple: A tuple containing the new forecast version (str) and the existing forecasts DataFrame (pd.DataFrame).
    
    Raises:
        Exception: If an error occurs while processing the existing forecast data.
    """
    try:
        # Read the existing forecast data from the CSV file
        old_forecasts = read_data_from_csv(forecast_output_table_path)
        if old_forecasts is None:
            logging.error(f"Unable to load the existing forecasts from {forecast_output_table_path}.")
            return None, None
        
        logging.info("Existing forecast data loaded successfully.")
        
        # Extract version numbers from the existing forecast data
        existing_versions = old_forecasts[old_forecasts['FORECAST_VERSION'].str.startswith(version_prefix)]['FORECAST_VERSION']
        version_numbers = existing_versions.str.extract(r'_(\d+)$')
        
        # Determine the next version number
        if version_numbers.empty:
            new_version = f"{version_prefix}01"
        else:
            max_version = version_numbers.astype(int).max()[0]
            new_version = f"{version_prefix}{str(max_version + 1).zfill(2)}"
        
        logging.info(f"New forecast version determined: {new_version}")
        return new_version, old_forecasts
    
    except Exception as e:
        logging.error(f"Error processing the existing forecast data: {e}")
        raise


def create_new_forecast_version(version_prefix):
    """
    Creates a new forecast version when the output table does not exist.
    
    Args:
        version_prefix (str): The prefix to use for the forecast version.
    
    Returns:
        str: The newly created forecast version.
    """
    new_version = f"{version_prefix}01"
    logging.info(f"New forecast version created: {new_version}")
    return new_version


def update_forecast_with_version(forecast_df, new_version):
    """
    Updates the forecast DataFrame with a new version and run ID.
    
    Args:
        forecast_df (pd.DataFrame): The forecast DataFrame to update.
        new_version (str): The new forecast version to assign.
    
    Returns:
        pd.DataFrame: The updated forecast DataFrame with version and run ID added.
    """
    # Add the new version and run ID to the forecast DataFrame
    forecast_df["FORECAST_VERSION"] = new_version
    forecast_df["RUN_ID"] = "RUN_" + forecast_df["FORECAST_VERSION"]
    logging.info("Forecast data updated with new version and run ID.")
    return forecast_df