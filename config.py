import torch

torch.manual_seed(42)

default_config = {
    'comet_api_key_file_path': './comet_api_key.txt',
    'device': 'cuda',
    'experiment_name': 'trials',
    'comet_project_name': 'arabic-did-debug',
    'datasets_dir': r'../datasets',
    'datasets': ['PADIC'],
    'training': {'train_batch_size': 32, 'n_train_workers': 8, 'log_every_n_batches': 50,
                  'training_epochs': 15, 'eval_every_n_batches' : 200},
    'tune' : {'discriminating_metric' : 'micro_average_accuracy', 'resources_per_trial' : {'cpu' : 2, 'gpu' : 1}, 'working_dir':r'../ray'},
    'evaluation' : {'metrics': ['micro_average_accuracy'],
                  'eval_batch_size': 32, 'n_eval_workers': 2},
    'penalize_all_steps': True,
    'optimizer': {'name': 'adam', 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08},
    'preprocessing': {'normalize': True, 'max_rep': 0, 'tokenization': 'char'},
    'model': 'simple_gru',
    'simple_gru': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.01, 'bidirectional': True,
                   'embedding_size': 128},
    'labels_to_int': {
        'MSA': 0,
        'ANN': 1,
        'MOR': 2,
        'SYR': 3,
        'PAL': 4,
        'ALG': 5}
}
