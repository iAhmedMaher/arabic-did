import os
import pandas as pd


def get_datasets_paths(config, type):
    paths = []
    for dataset_name in config['datasets']:
        paths.append(os.path.join(config['datasets_dir'], dataset_name, type, type + '.csv'))
    return paths

def get_combined_dataframes(csv_paths_list):
    print('Loading datasets ...')
    combined_dataset = None
    for csv_path in csv_paths_list:
        current_dataset = pd.read_csv(csv_path, index_col='id')
        combined_dataset = pd.concat([combined_dataset, current_dataset], axis=0)
        print('Dataset', csv_path, 'has been loaded.')

    print('All datasets are loaded.')

    return combined_dataset