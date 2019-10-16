import torch.nn as nn

def get_model(input_size, config):
    if config['model'] == 'simple_gru':
        return SimpleGRU(input_size,config)

    raise NotImplementedError()

class SimpleGRU(nn.Module):
    def __init__(self, input_size, config):
        super(SimpleGRU, self).__init__()
        self.embedding_layer = nn.Embedding(input_size, int(config['simple_gru']['embedding_size']), padding_idx=0, sparse=True)
        self.rnn = nn.GRU(int(config['simple_gru']['embedding_size']), int(config['simple_gru']['hidden_size']),
                          int(config['simple_gru']['num_layers']), dropout=config['simple_gru']['dropout'],
                          bidirectional=config['simple_gru']['bidirectional'])

        self.output_all = config['penalize_all_steps']
        num_directions = 2 if config['simple_gru']['bidirectional'] else 1
        self.output_scores = nn.Linear(in_features=num_directions*int(config['simple_gru']['hidden_size']),
                                       out_features=len(config['labels_to_int']))

    def get_non_sparse_parameters(self):
        return list(set(self.parameters()) - set(self.embedding_layer.parameters()))

    def get_sparse_parameters(self):
        return self.embedding_layer.parameters()

    def forward(self, x):
        x = self.embedding_layer(x)
        x, h_n = self.rnn(x)
        if not self.output_all or not self.training:
            x = x[-1, :, :].unsqueeze(0)
        x = self.output_scores(x)
        return x