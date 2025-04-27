import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

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
