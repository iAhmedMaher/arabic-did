import torch

torch.manual_seed(42)

default_config = {
    'comet_api_key_file_path': './comet_api_key.txt',
    'device': 'cuda',
    'experiment_name': 'debug',
    'comet_project_name': 'arabic-did-debug',
    'datasets_dir': r'../datasets',
    'datasets': ['PADIC'],
    'training': {'train_batch_size': 128, 'n_train_workers': 8, 'log_every_n_batches': 10,
                 'training_epochs': 15, 'eval_every_n_batches': 100},
    'tune': {'tuning_method': 'hyperopt', 'discriminating_metric': 'micro_average_accuracy',
             'discriminating_metric_mode': 'max', 'max_t': 10000, 'n_samples': 10000,
             'resources_per_trial': {'cpu': 2, 'gpu': 1}, 'working_dir': '../ray', 'resume':True},
    'evaluation': {'metrics': ['per_class_precision', 'per_class_recall', 'per_class_f1', 'micro_average_accuracy',
                               'macro_average_precision', 'macro_average_recall', 'macro_average_f1', 'eval_loss', 'in_out', 'cm'],
                   'eval_batch_size': 32, 'n_eval_workers': 2},
    'penalize_all_steps': True,
    'optimizer': {'name': 'adam', 'lr': 0.005, 'betas': (0.9, 0.999), 'eps': 1e-08},
    'preprocessing': {'normalize': True, 'max_rep': 0, 'tokenization': 'word', 'per_class_vocab_size' : 2000},
    'model': 'simple_gru',
    'simple_gru': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.005, 'bidirectional': True,
                   'embedding_size': 128},
    'labels_to_int': {
        'MSA': 0,
        'ANN': 1,
        'MOR': 2,
        'SYR': 3,
        'PAL': 4,
        'ALG': 5}
}
