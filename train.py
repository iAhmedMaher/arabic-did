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
import collections
from eval import Evaluation
import numpy as np
import random

train_dataset = None
eval_dataset = None


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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
                      batch_size=config['training']['train_batch_size'],
                      shuffle=True,
                      drop_last=False,
                      num_workers=config['training']['n_train_workers'],
                      collate_fn=transformer)


def get_eval_dataloader(config, transformer):
    global eval_dataset

    if eval_dataset is None:
        eval_paths = get_datasets_paths(config, 'eval')
        eval_dataset = data.CSVDatasetsMerger(eval_paths)

    return DataLoader(eval_dataset,
                      batch_size=config['evaluation']['eval_batch_size'],
                      shuffle=False,
                      drop_last=False,
                      num_workers=config['evaluation']['n_eval_workers'],
                      collate_fn=transformer)


def get_optimizers(model, config):
    if config['optimizer']['name'] == 'adam':
        non_sparse = optim.Adam(model.get_non_sparse_parameters(), lr=config['optimizer']['lr'],
                                betas=config['optimizer']['betas'],
                                eps=config['optimizer']['eps'])
        sparse = optim.SparseAdam(model.get_sparse_parameters(), lr=config['optimizer']['lr'],
                                  betas=config['optimizer']['betas'],
                                  eps=config['optimizer']['eps'])
        return non_sparse, sparse
    else:
        raise NotImplementedError()


def setup_training(config):
    experiment = Experiment(get_comet_api_key(config), project_name=config['comet_project_name'], log_code=True)
    experiment.log_parameters(flatten_dict(config))

    text_proc = TextPreprocessor(config)

    train_dataloader = get_train_dataloader(config, text_proc)
    eval_dataloader = get_eval_dataloader(config, text_proc)

    model = m.get_model(text_proc.get_num_tokens(), config)
    return experiment, model, train_dataloader, eval_dataloader


def normal_training(config):
    device = torch.device(config['device'])
    print('Using device', device)
    exp, model, train_dataloader, eval_dataloader = setup_training(config)
    exp.set_name(config['experiment_name'])
    model.train()
    model = model.to(device)
    optimizers = get_optimizers(model, config)
    evaluator = Evaluation(eval_dataloader, config)

    num_examples = 0
    for epoch in range(config['training']['training_epochs']):
        for idx, batch in enumerate(train_dataloader):
            batch = (batch[0].to(device), batch[1].to(device))
            num_examples += len(batch[0])
            loss = training_step(batch, model, optimizers)
            if idx % config['training']['log_every_n_batches'] == 0:
                print(epoch, num_examples, loss.detach().cpu().numpy())
                exp.log_metric('train_loss', loss.detach().cpu().numpy(), step=num_examples, epoch=epoch)

            if idx % config['training']['eval_every_n_batches'] == 0:
                results = evaluator.eval_model(model)
                for metric in results:
                    print(metric, results[metric])
                    exp.log_metric(metric, results[metric], step=num_examples, epoch=epoch)


def training_step(training_batch, model, optimizers):
    [opt.zero_grad() for opt in optimizers]
    labels, tokens = training_batch
    outputs = model(tokens)
    output_len, classes = outputs.size()[0], outputs.size()[2]
    outputs = outputs.view(-1, classes)
    labels = labels.repeat(output_len)
    loss = cross_entropy(outputs, labels)
    loss.backward()
    [opt.step() for opt in optimizers]
    return loss


def tune_training(config):
    from hyper_tune import TuneTrainable
    import ray
    from ray import tune
    from ray.tune.schedulers import HyperBandScheduler
    from ray.tune import sample_from

    ray.init()

    scheduler = HyperBandScheduler(time_attr='num_examples', metric=config['tune']['discriminating_metric'], mode='max',
                                   max_t=300000)

    config['simple_gru']['hidden_size'] = sample_from(lambda spec: 2 ** random.randint(4, 8))
    tune.run(TuneTrainable, scheduler=scheduler, config=config, num_samples=3, name=config['experiment_name'],
             resources_per_trial=config['tune']['resources_per_trial'], local_dir=config['tune']['working_dir'])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    tune_training(cfg.default_config)
