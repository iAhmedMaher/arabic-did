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
    from ray.tune.schedulers import HyperBandScheduler, HyperBandForBOHB, AsyncHyperBandScheduler
    import ray.tune.suggest as suggest
    from ray.tune import sample_from
    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
    from ray.tune.suggest.hyperopt import HyperOptSearch
    from hyperopt import hp

    ray.init()

    if config['tune']['tuning_method'] == 'bohb':
        config_space = CS.ConfigurationSpace(seed=42)

        # replace | convention is a kludge because of BOHB's specialized interface
        config_space.add_hyperparameters([CSH.UniformIntegerHyperparameter('replace|num_layers', lower=1, upper=5),
                                          CSH.UniformIntegerHyperparameter('replace|hidden_size', lower=64, upper=512),
                                          CSH.UniformIntegerHyperparameter('replace|embedding_size', lower=64,
                                                                           upper=512),
                                          CSH.UniformFloatHyperparameter('replace|dropout', lower=0.0, upper=0.5),
                                          CSH.CategoricalHyperparameter('replace|bidirectional', choices=[True, False]),
                                          CSH.UniformFloatHyperparameter('replace|lr', lower=0.00001, upper=0.1,
                                                                         log=True)])
        bohb_hyperband = HyperBandForBOHB(time_attr='num_examples', metric=config['tune']['discriminating_metric'],
                                          mode=config['tune']['discriminating_metric_mode'],
                                          max_t=config['tune']['max_t'])

        bohb_search = suggest.bohb.TuneBOHB(config_space, max_concurrent=1,
                                            metric=config['tune']['discriminating_metric'],
                                            mode=config['tune']['discriminating_metric_mode'])

        return tune.run(TuneTrainable, config=config, scheduler=bohb_hyperband, search_alg=bohb_search, num_samples=1,
                        name=config['experiment_name'],
                        resources_per_trial=config['tune']['resources_per_trial'],
                        local_dir=config['tune']['working_dir'])

    elif config['tune']['tuning_method'] == 'hyperopt':
        space = {"replace|hidden_size": hp.quniform("replace|hidden_size", 64, 512, 2),
                 "replace|embedding_size": hp.quniform("replace|embedding_size", 128, 1024, 2),
                 "replace|bidirectional": hp.choice("replace|bidirectional", [True, False]),
                 "replace|num_layers": hp.quniform("replace|num_layers", 1, 5, 1),
                 "replace|dropout": hp.loguniform("replace|dropout", -11, -0.8)}
        algo = HyperOptSearch(space, max_concurrent=10, metric=config['tune']['discriminating_metric'],
                              mode=config['tune']['discriminating_metric_mode'])
        scheduler = HyperBandScheduler(time_attr='num_examples', metric=config['tune']['discriminating_metric'],
                                       mode=config['tune']['discriminating_metric_mode'],
                                       max_t=config['tune']['max_t'])
        return tune.run(TuneTrainable, config=config, scheduler=scheduler, search_alg=algo, num_samples=10,
                        name=config['experiment_name'],
                        resources_per_trial=config['tune']['resources_per_trial'],
                        local_dir=config['tune']['working_dir'])

    else:
        raise NotImplementedError()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    print(tune_training(cfg.default_config))
