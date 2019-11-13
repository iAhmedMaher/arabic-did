from comet_ml import Experiment
import torch
from preprocessing import TextPreprocessor
import config as cfg
import helpers
import models
import os
import data
from torch.utils.data import DataLoader
from eval import Evaluation
from train import get_comet_api_key  # TODO refactor
import uuid


if __name__ == '__main__':
    print('Starting testing ...')
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    config = cfg.default_config

    exp = Experiment(get_comet_api_key(config), project_name=config['comet_project_name'], log_code=True)
    exp.set_name(config['experiment_name'] + uuid.uuid4().hex)

    train_df = helpers.get_combined_dataframes(helpers.get_datasets_paths(config, 'train'))

    text_proc = TextPreprocessor(config, train_df, return_text=True)

    print('Loading model ...')
    model = models.get_model(text_proc.n_tokens, config)
    model.to(config['device'])
    checkpoint = torch.load(os.path.join('checkpoints', config['load_checkpoint']), map_location=torch.device(config['device']))
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded.')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_dict = helpers.flatten_dict(config)
    log_dict.update({'trainable_params': n_params})
    exp.log_parameters(log_dict)

    test_dataset = data.CSVDatasetsMerger(helpers.get_datasets_paths(config, 'test'))
    test_dataloader = DataLoader(test_dataset,
                       batch_size=config['evaluation']['eval_batch_size'],
                       shuffle=False,
                       drop_last=False,
                       num_workers=config['evaluation']['n_eval_workers'],
                       collate_fn=text_proc)

    evaluator = Evaluation(test_dataloader, config)

    print('Testing ...')
    results, assets, image_fns = evaluator.eval_model(model, finished_training=True)
    print('Finished testing. Uploading ...')

    exp.log_metrics(results, step=0, epoch=0)
    [exp.log_asset_data(asset, step=0) for asset in assets]
    [exp.log_image(fn, step=0) for fn in image_fns]

    print('Finished uploading.')




