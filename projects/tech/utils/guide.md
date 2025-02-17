# Apply multiprocessing in pandas
```python
import multiprocessing
import time
import requests
import pandas as pd

def pandas_api_call(url):
    """
    Fetches a single record from the API with retry logic.

    Args:
        url: The API endpoint URL.

    Returns:
        The fetched record or an error message.
    """
    max_retries = 3
    delay = 60

    for attempt in range(max_retries):
        try:
            response = requests.get(url)

            if response.status_code == 200:
                #return response.json()["random_record"]
                return pd.Series(["summary","success"])

            print(f"Attempt {attempt+1}: Error {response.status_code}, retrying in {delay} seconds...")
            time.sleep(delay)

        except Exception as e:
            print(f"Attempt {attempt+1}: Exception {str(e)}, retrying in {delay} seconds...")
            time.sleep(delay)
            

    #return "API request failed after multiple attempts."
    return pd.Series(["no resp","failed"])


def process_row(row):
    """
    Calls the API for the given row (not used in this simplified example).
    """
    # This function would typically use the row data to construct the API URL
    # and then call the api_call function.
    # For this example, we'll assume a constant URL:
    return pandas_api_call(API_URL) 

# Assuming dfp is your pandas DataFrame
with multiprocessing.Pool(processes=10) as pool:  # Create a pool of n worker processes
    #dfp["pandas_req"] = dfp.apply(process_row, axis=1)
    #dfp["pandas_req"] = pool.map(process_row, dfp.to_dict(orient="records"))
    results = pool.map(process_row, dfp.iterrows())
    
dfp[["summary", "status"]] = pd.DataFrame(results, index=dfp.index)
```
# How to log?
``` python
import logging
import time
import argparse

# # Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('mylogs.log')])

# Dummy paths
raw_data_path = '/path/to/raw/data'
processed_data_path = '/path/to/processed/data'
model_path = '/path/to/model'

def log_paths(raw_path, processed_path, model_path,version):
    logging.info(f'==================================================================')
    logging.info('\n'*2)
    logging.info(f'experiemnt_v{version}')
    logging.info(f'Raw Data Path: {raw_path}_v{version}')
    time.sleep(5)    
    logging.info(f'Processed Data Path: {processed_path}_v{version}')
    time.sleep(5)
    logging.info(f'Model Path: {model_path}_v{version}')
    logging.info('\n'*2)
    logging.info(f'==================================================================')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some data paths.') 
    parser.add_argument('--version', type=str, required=True, help='Version number') 
    args = parser.parse_args()
    version = args.version
    log_paths(raw_data_path, processed_data_path, model_path,version)
```
