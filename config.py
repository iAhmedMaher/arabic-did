import torch
torch.manual_seed(42)

default_config = {
    'datasets_dir': '..\datasets',
    'datasets': ['PADIC', 'PADIC'],
    'train_batch_size': 4,
    'n_train_workers': 1,
    'eval_batch_size': 4,
    'n_eval_workers': 1,
    'preprocessing': {'normalize': True, 'max_rep': 0, 'tokenization': 'char', 'max_seq_len': 150},
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
