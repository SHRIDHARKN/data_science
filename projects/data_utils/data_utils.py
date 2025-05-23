import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from termcolor import colored

# === LOG ===
def log_msg(msg, msg_typ="normal", font_color="white", bg_color="red", header="", sep = "="):
    
    msg = " \u039B | " +" "*2+ msg +" "*10
    
    if msg_typ == "normal":
        print(colored(sep * 10 + " INFO " + sep * 10, 'black', 'on_white', attrs=['bold'])) 
        print(colored(msg, 'black', 'on_white', attrs=['bold']))   
    elif msg_typ == "data":
        print(colored(sep * 10 + " DATA/ VARS / CONFIG " + sep * 10, 'black', 'on_white', attrs=['bold'])) 
        print(colored(msg, 'white', 'on_blue', attrs=['bold']))
    elif msg_typ == "progress":
        print(colored(sep * 10 + " PROGRESS/ SAVES " + sep * 10, 'black', 'on_white', attrs=['bold'])) 
        print(colored(msg, 'white', 'on_green', attrs=['bold']))
    else:
        print(colored(sep * 10 + header + sep * 10, 'black', 'on_white', attrs=['bold'])) 
        print(colored(msg, font_color, f'on_{bg_color}', attrs=['bold']))
    print(colored(sep * 50+"\n", 'black', 'on_white', attrs=['bold'])) 


# === LOAD DATA ===
class CSVDataset(Dataset):
    def __init__(self, root_dir):
        
        self.root_dir = root_dir
        self.csv_files = [f for f in os.listdir(root_dir) if f.endswith('.csv')]
        self.all_data = []
        self._load_metadata()
        self.num_files = len(self.csv_files)
        print(f"Found {self.num_files} CSV files in {self.root_dir}")

    def _load_metadata(self):
        
        for file_idx, filename in enumerate(self.csv_files):
            filepath = os.path.join(self.root_dir, filename)
            df = pd.read_csv(filepath, header=0)  # Assuming your CSV has a header row
            for row_idx in range(len(df)):
                self.all_data.append((file_idx, row_idx))
                
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        file_idx, row_idx = self.all_data[idx]
        filename = self.csv_files[file_idx]
        filepath = os.path.join(self.root_dir, filename)
        df = pd.read_csv(filepath, header=0)
        sample = df.iloc[row_idx].to_dict()  # Get the row as a dictionary
        return sample

class ParquetDataset(Dataset):
    def __init__(self, root_dir):
        
        self.root_dir = root_dir
        self.parquet_files = [f for f in os.listdir(root_dir) if f.endswith('.parquet')]
        self.all_data = []
        self._load_metadata()
        self.num_files = len(self.parquet_files)
        print(f"Found {self.num_files} parquet files in {root_dir}")

    def _load_metadata(self):
        
        for file_idx, filename in enumerate(self.parquet_files):
            filepath = os.path.join(self.root_dir, filename)
            df = pd.read_parquet(filepath)
            for row_idx in range(len(df)):
                self.all_data.append((file_idx, row_idx))
                
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        file_idx, row_idx = self.all_data[idx]
        filename = self.parquet_files[file_idx]
        filepath = os.path.join(self.root_dir, filename)
        df = pd.read_parquet(filepath)
        sample = df.iloc[row_idx].to_dict()  # Get the row as a dictionary
        return sample
