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
