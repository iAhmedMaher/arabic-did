import torch
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.manual_seed(42)


default_config = {
    'comet_api_key_file_path': './comet_api_key.txt',
    'device': 'cuda',
    'experiment_name': 'char-lstm',
    'comet_project_name': 'arabic-did-debug',
    'datasets_dir': r'../datasets',
    'datasets': ['Shami'],
    'training': {'train_batch_size': 128, 'n_train_workers': 8, 'log_every_n_batches': 10,
                 'training_epochs': 15, 'eval_every_n_batches': 100, 'shuffle_train_eval' : True},
    'tune': {'tuning_method': 'no_search', 'discriminating_metric': 'micro_average_accuracy',
             'discriminating_metric_mode': 'max', 'max_t': 250000, 'n_samples': 1,
             'resources_per_trial': {'cpu': 2, 'gpu': 1}, 'working_dir': '../ray', 'resume': True},
    'evaluation': {'metrics': ['per_class_precision', 'per_class_recall', 'per_class_f1', 'micro_average_accuracy',
                               'macro_average_precision', 'macro_average_recall', 'macro_average_f1', 'eval_loss', 'in_out', 'cm'],
                   'eval_batch_size': 128, 'n_eval_workers': 8},
    'penalize_all_steps': True,
    'optimizer': {'name': 'adam', 'lr': 0.005, 'betas': (0.9, 0.999), 'eps': 1e-08},
    'preprocessing': {'normalize': True, 'max_rep': 0, 'tokenizer': 'standard_tokenizer', 'max_seq_len' : 200},
    'model': 'simple_lstm',
    'simple_gru': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.005, 'bidirectional': True,
                   'embedding_size': 128},
    'simple_lstm': {'hidden_size': 394, 'num_layers': 2, 'dropout': 0.005, 'bidirectional': False,
                   'embedding_size': 490},
    'standard_tokenizer' : {'tokenization':'char', 'per_class_vocab_size' : 2000},
    'transformers_tokenizer' : {'pretrained' : 'bert-base-multilingual-uncased'},
    'bert' : {'pretrained' : 'bert-base-multilingual-uncased'},
    'labels_to_int': {
        'PAL': 0,
        'LEB': 1,
        'JOR': 2,
        'SYR': 3}
}
