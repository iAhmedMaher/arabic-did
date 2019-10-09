import torch.nn as nn


class SimpleGRU(nn.Module):
    def __init__(self, input_size, config):
        super(SimpleGRU, self).__init__()
        self.embedding_layer = nn.Embedding(input_size, config['embedding_size'], padding_idx=0)
        self.rnn = nn.GRU(config['embedding_size'], config['hidden_size'],
                          config['num_layers'], dropout=config['dropout'],
                          bidirectional=config['bidirectional'])

        # TODO: output should be calculated on last step only (depending on loss function)
        num_directions = 2 if config['bidirectional'] else 1
        self.output_scores = nn.Linear(in_features=num_directions*config['hidden_size'],
                                       out_features=len(config['labels_to_int']))
        self.output_probs = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.embedding_layer(x)
        x, h_n = self.rnn(x)
        x = self.output_scores(x)
        return self.output_probs(x)