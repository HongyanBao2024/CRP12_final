'''
model pipeline 
first run all models (S1-S3), followed with S4 mixing 
'''
import subprocess
import time
import pandas as pd
import copy
import os
import warnings
import sys
# sys.path.append('/mnt/batch/tasks/shared/LS_root/mounts/clusters/zcao-dev2/code/Users/zcao/artc_codebase/')
sys.path.append('/mnt/c/Users/baohy/Documents/VSProgram/CRP12/')
warnings.filterwarnings('ignore')
from utils.helper_functions import load_yaml_config

def run_script(env_name, script_name, *args):
    """
    Activate a conda environment and run a Python script in it.

    Parameters:
    - env_name: str, name of the conda environment
    - script_name: str, path to the Python script
    - *args: additional arguments to pass to the script
    """
    try:
        print(f"=========================Trying to execute {script_name} in {env_name}")
        # Use a properly initialized shell to activate the conda environment and run the script
        command = (
            f"source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda activate {env_name} && "
            f"python {script_name} {' '.join(map(str, args))}"
        )
        print(f"Running: {command}")
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
        print(f"=========================Successful execution of {script_name} in {env_name}")
    except subprocess.CalledProcessError as e:
        print(f"=================================Error occurred while running {script_name} in {env_name}: {e}")
        # break


def tasks(
        horizon,# =13,  # horizon
        LAG,# =2,  # lag weeks
        context_size,# =128,  # context
        freq,# ="W",  # freq
        testing_len,# =13,  # testing_len
        overwrite_flag,
        task,
        identifier,
        demand_data_path,
        initial_processing):

    task_stages = [
        #s1
        {"env": "crp12_nf",
         "script": "S1_neuralforecast.py",
         "args": [horizon,  # horizon
                  LAG,  # lag weeks
                  context_size,  # context
                  freq,  # freq
                  testing_len,  # testing_len
                  overwrite_flag,  # overwrite_flag
                  task,
                  identifier,
                  demand_data_path,
                  initial_processing
                  ]},  # Replace arg1, arg2 with actual arguments

        #s2
        {"env": "tfm_env",
         "script": "S2_tfm.py",
         "args": [horizon,
                  context_size,
                  testing_len,
                  identifier]},
        
        #s3
        {"env": "uni2ts",
         "script": "S3_uni2ts.py",
         "args": [horizon,
                  context_size,
                  testing_len,
                  identifier,
                  LAG]}]

    # Execute each task

    for task in task_stages:
        print(f"=======================executing task {task['script']}==========================")
        run_script(task["env"], task["script"], *task["args"])


def main():
    # load config
    processing_config_path = "configs/forecast_pipeline_config.yaml"
    config = load_yaml_config(processing_config_path)
    print(config)
    # Access parameters from the YAML file
    context_size = config["parameters"]["context_size"]
    horizon = config["parameters"]["horizon"]
    LAG = config["parameters"]["LAG"]
    freq = config["parameters"]["freq"]
    testing_len = config["parameters"]["testing_len"]
    overwrite_flag = config["parameters"]["overwrite_flag"]
    task = config["parameters"]["task"]
    initial_processing = config["parameters"]["initial_processing"]
    demand_data_path = config["paths"]["demand_data_path"]

    # context_size: 24 # test use 32 real use 128
    # horizon: 12 # test use 4 real use 13
    # LAG: 3
    # freq: "M"
    # testing_len: 4 # test use 4 real use 13
    # overwrite_flag: True
    # task: 'TCCC_sg'

    # identifier = f'/mnt/c/data/CRP12/KC_monthly/{task}_context_size_{context_size}_horizon_{horizon}_testing_len_{testing_len}_KC_Lag3_deep_robust_all'
    identifier = f'inputoutput/KC_monthly/{task}_context_size_{context_size}_horizon_{horizon}_testing_len_{testing_len}_KC_Lag3'

    start_time = time.time()
    tasks(
        horizon=horizon,
        LAG=LAG,
        context_size=context_size,
        freq=freq,
        testing_len=testing_len,
        overwrite_flag=overwrite_flag,
        task = task,
        identifier=identifier,
        demand_data_path = demand_data_path,
        initial_processing = initial_processing
    )
    
    
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Time taken for task: {run_time:.2f} seconds")

if __name__ == "__main__":
    main()
