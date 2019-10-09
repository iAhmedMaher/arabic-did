from torch.utils.data import Dataset
import pandas as pd
import config as cfg

class CSVDatasetsMerger(Dataset):
    """
    Assumes all datasets are standardized
    """
    def __init__(self, csv_paths_list):
        print('Loading datasets ...')
        self.combined_dataset = None
        for csv_path in csv_paths_list:
            current_dataset = pd.read_csv(csv_path, index_col='id')
            self.combined_dataset = pd.concat([self.combined_dataset, current_dataset], axis=0)
            print('Dataset', csv_path, 'has been loaded.')

        print('All datasets are loaded.')

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        item = self.combined_dataset.iloc[idx, :]
        label, text = item['label'], item['text']
        return label, text




