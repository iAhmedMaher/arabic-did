from torch.utils.data import DataLoader, Dataset
import os
import data
from preprocessing import TextPreprocessor
import config as cfg

train_dataset = None
eval_dataset = None

def get_datasets_paths(config, type):
    paths = []
    for dataset_name in config['datasets']:
        paths.append(os.path.join(config['datasets_dir'], dataset_name, type, type+'.csv'))
    return paths

def get_train_dataloader(config):
    global train_dataset

    if train_dataset is None:
        train_paths = get_datasets_paths(config, 'train')
        train_dataset = data.CSVDatasetsMerger(train_paths)

    transformer = TextPreprocessor(config)
    return DataLoader(train_dataset,
                      batch_size=config['train_batch_size'],
                      shuffle=True,
                      drop_last=False,
                      num_workers=config['n_train_workers'],
                      collate_fn=transformer)

def get_eval_dataloader(config):
    global eval_dataset

    if eval_dataset is None:
        eval_paths = get_datasets_paths(config, 'eval')
        eval_dataset = data.CSVDatasetsMerger(eval_paths)

    transformer = TextPreprocessor(config)
    return DataLoader(eval_dataset,
                      batch_size=config['eval_batch_size'],
                      shuffle=False,
                      drop_last=False,
                      num_workers=config['n_eval_workers'],
                      collate_fn=transformer)


if __name__ == '__main__':
    train = get_train_dataloader(cfg.default_config)
    eval = get_eval_dataloader(cfg.default_config)

    for i, b in enumerate(train):
        if i == 10:
            for be in eval:
                pass

