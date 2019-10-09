default_config = {
    'datasets_dir' : '..\datasets',
    'datasets' : ['PADIC', 'PADIC'],
    'train_batch_size' : 4,
    'n_train_workers' : 1,
    'eval_batch_size' : 4,
    'n_eval_workers' : 1,
    'preprocessing' : {'normalize' : True, 'max_rep' : 0, 'tokenization' : 'char', 'max_seq_len' : 150 },

}