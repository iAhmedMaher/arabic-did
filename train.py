from comet_ml import Experiment
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import data
from preprocessing import TextPreprocessor
import config as cfg
import models as m
import torch

train_dataset = None
eval_dataset = None


def get_comet_api_key(config):
    path = config['comet_api_key_file_path']
    with open(path, 'r') as f:
        return f.readline().rstrip()


def get_datasets_paths(config, type):
    paths = []
    for dataset_name in config['datasets']:
        paths.append(os.path.join(config['datasets_dir'], dataset_name, type, type + '.csv'))
    return paths


def get_train_dataloader(config, transformer):
    global train_dataset

    if train_dataset is None:
        train_paths = get_datasets_paths(config, 'train')
        train_dataset = data.CSVDatasetsMerger(train_paths)

    return DataLoader(train_dataset,
                      batch_size=config['train_batch_size'],
                      shuffle=True,
                      drop_last=False,
                      num_workers=config['n_train_workers'],
                      collate_fn=transformer)


def get_eval_dataloader(config, transformer):
    global eval_dataset

    if eval_dataset is None:
        eval_paths = get_datasets_paths(config, 'eval')
        eval_dataset = data.CSVDatasetsMerger(eval_paths)

    return DataLoader(eval_dataset,
                      batch_size=config['eval_batch_size'],
                      shuffle=False,
                      drop_last=False,
                      num_workers=config['n_eval_workers'],
                      collate_fn=transformer)


def get_optimizer(model, config):
    if config['optimizer']['name'] == 'sparse_adam':
        return optim.Adam(model.parameters(), lr=config['optimizer']['lr'], betas=config['optimizer']['betas'],
                                eps=config['optimizer']['eps'])
    else:
        raise NotImplementedError()

def setup_training(config):
    experiment = Experiment(get_comet_api_key(config), project_name=config['comet_project_name'])
    experiment.log_parameters(config['model'])

    text_proc = TextPreprocessor(config)

    train_dataloader = get_train_dataloader(config, text_proc)
    eval_dataloader = get_eval_dataloader(config, text_proc)

    model = m.get_model(text_proc.get_num_tokens(), config)
    return experiment, model, train_dataloader, eval_dataloader


def normal_training(config):
    device = torch.device(config['device'])
    print('Using device', device)
    exp, model, train_dataloader, eval_dataloader = setup_training(config)
    model.train()
    model = model.to_device(device)
    optimizer = get_optimizer(model, config)
    epoch = 0
    while True:
        for idx, batch in enumerate(train_dataloader):
            batch = batch.to_device(device)
            loss = training_step(batch, model, optimizer)
            if idx % 50 == 0:
                print(epoch, idx, loss.detach().numpy())
                exp.log_metric('loss', loss.detach().numpy(), step=idx + len(train_dataloader) * epoch, epoch=epoch)
        epoch += 1


def training_step(training_batch, model, optimizer):
    optimizer.zero_grad()
    labels, tokens = training_batch
    outputs = model(tokens)
    output_len, classes = outputs.size()[0], outputs.size()[2]
    outputs = outputs.view(-1, classes)
    labels = labels.repeat(output_len)
    loss = cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    print(os.getcwd())
    normal_training(cfg.default_config)
