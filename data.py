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

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        item = self.combined_dataset.iloc[idx, :]
        label, text = item['label'], item['text']
        return label, text

    def get_pandas_df(self):
        return self.combined_dataset


class PandasDataset(Dataset):
    def __init__(self, pandas_df):
        """
    Assumes dataframe is standardized
    """
        self.pandas_df = pandas_df

    def __len__(self):
        return len(self.pandas_df)

    def __getitem__(self, idx):
        item = self.pandas_df.iloc[idx, :]
        label, text = item['label'], item['text']
        return label, text

    def get_pandas_df(self):
        return self.pandas_df




