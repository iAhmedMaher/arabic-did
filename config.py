import torch
import random
import numpy as np

# random.seed(42)
# np.random.seed(42)
torch.manual_seed(42)

default_config = {
    'comet_api_key_file_path': './comet_api_key.txt',
    'device': 'cuda',
    'experiment_name': 'eval-vdcnn-paper',
    'comet_project_name': 'arabic-did-paper',
    'datasets_dir': r'../datasets',
    'datasets': ['Shami'],
    'training': {'back_prop_every_n_batches': 1, 'train_batch_size': 80, 'n_train_workers': 8, 'log_every_n_batches': 10,
                 'training_epochs': 15, 'eval_every_n_batches': 100, 'shuffle_train_eval': False, 'checkpoint_best' : True, 'dataset_size' : 1.0},
    'tune': {'tuning_method': 'no_search', 'discriminating_metric': 'micro_average_accuracy',
             'discriminating_metric_mode': 'max', 'max_t': 1000000, 'n_samples': 2,
             'resources_per_trial': {'cpu': 2, 'gpu': 1}, 'working_dir': '../ray', 'resume': False},
    'evaluation': {'metrics': ['per_class_precision', 'per_class_recall', 'per_class_f1', 'micro_average_accuracy',
                               'macro_average_precision', 'macro_average_recall', 'macro_average_f1', 'eval_loss',
                               'in_out', 'cm'],
                   'eval_batch_size': 80, 'n_eval_workers': 8},
    'penalize_all_steps': True,
    'optimizer': {'name': 'adam', 'lr': 1.765380655814118E-4, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 2.185363571888231E-9},
    'preprocessing': {'normalize': True, 'max_rep': 0, 'tokenizer': 'standard_tokenizer', 'max_seq_len': 300},
    'model': 'vdcnn',
    'load_checkpoint': 'eval-awd-lstm9c9e86e2006b4386b9d9662df34b0427.pt',
    'simple_gru': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.005, 'bidirectional': True,
                   'embedding_size': 128},
    'simple_lstm': {'hidden_size': 394, 'num_layers': 2, 'dropout': 0.005, 'bidirectional': False,
                    'embedding_size': 490},
    'awd_rnn': {'hidden_size': 608, 'num_layers': 4, 'dropouth': 0.33342707216370937, 'dropouti': 0.20239988986112784, 'dropoute': 0.024969153615953168, 'wdrop': 0,
                'dropouto': 0.0, 'ar_alpha': 6.782895014034089,
                'embedding_size': 492},
    'vdcnn': {'embedding_size': 416, 'dropout': 0.09100404232504053, 'k': 6, 'conv1_nblocks': 2, 'conv2_nblocks': 1, 'conv3_nblocks': 1,
              'conv4_nblocks': 2,
              'conv0_nfmaps': 135, 'conv1_nfmaps': 58, 'conv2_nfmaps': 103, 'conv3_nfmaps': 177, 'conv4_nfmaps': 543,
              'dense_nlayers': 1, 'dense_nfeatures': 3515, 'apply_shortcut': True},
    'bert' : {'hidden_size' : 224, 'n_bert_layers': 2, 'n_att_heads':7, 'intermediate_dense_size':852, 'hidden_dropout':0.4362316177107697, 'att_dropout':0.0239794459965727},
    'standard_tokenizer': {'tokenization': 'word', 'per_class_vocab_size': 6900},
    'youtokentome': {'vocab_size': 31400},
    'transformers_tokenizer': {'model': 'bert-base-multilingual-uncased'},
    'bert_pretrained': {'model': 'bert-base-multilingual-uncased'},
    'labels_to_int': {
        'PAL': 0,
        'LEB': 1,
        'JOR': 2,
        'SYR': 3}
}
