from torch.utils.data import Dataset
import pandas as pd
import config as cfg
from helpers import get_combined_dataframes

# TODO refactor

class CSVDatasetsMerger(Dataset):
    """
    Assumes all datasets are standardized
    """
    def __init__(self, csv_paths_list):
        self.combined_dataset = get_combined_dataframes(csv_paths_list)
        self.served_df = self.combined_dataset

    def __len__(self):
        return len(self.served_df)

    def __getitem__(self, idx):
        item = self.served_df.iloc[idx, :]
        label, text = item['label'], item['text']
        return label, text

    def get_pandas_df(self):
        return self.served_df

    def limit_dataset_size(self, percent):
        new_size = int(len(self.combined_dataset) * percent)
        self.served_df = self.combined_dataset.loc[:new_size, :]


class PandasDataset(Dataset):
    def __init__(self, pandas_df):
        """
    Assumes dataframe is standardized
    """
        self.pandas_df = pandas_df
        self.served_df = self.pandas_df

    def __len__(self):
        return len(self.served_df)

    def __getitem__(self, idx):
        item = self.served_df.iloc[idx, :]
        label, text = item['label'], item['text']
        return label, text

    def get_pandas_df(self):
        return self.served_df

    def limit_dataset_size(self, percent):
        new_size = int(len(self.pandas_df) * percent)
        self.served_df = self.pandas_df.loc[:new_size, :]


