import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig
import torch


def get_model(input_size, config):
    if config['model'] == 'simple_gru':
        return SimpleGRU(input_size, config)
    elif config['model'] == 'simple_lstm':
        return SimpleLSTM(input_size, config)
    elif config['model'] == 'bert':
        return Bert(input_size, config)

    raise NotImplementedError()


class Bert(nn.Module):
    def __init__(self, input_size, config):
        super(Bert, self).__init__()
        self.transformer = BertForSequenceClassification.from_pretrained(config['bert']['pretrained'],
                                                                         num_labels=len(config['labels_to_int']))

    def forward(self, x):
        x = x.permute(1,0)
        attn_mask = (x != 0).float()
        x = self.transformer(x, attention_mask=attn_mask)[0].unsqueeze(0)
        return x

    def get_non_sparse_parameters(self):
        return self.parameters()

    def get_sparse_parameters(self):
        return []


class SimpleGRU(nn.Module):
    def __init__(self, input_size, config):
        super(SimpleGRU, self).__init__()
        self.embedding_layer = nn.Embedding(input_size, int(config['simple_gru']['embedding_size']), padding_idx=0,
                                            sparse=True)
        self.rnn = nn.GRU(int(config['simple_gru']['embedding_size']), int(config['simple_gru']['hidden_size']),
                          int(config['simple_gru']['num_layers']), dropout=abs(config['simple_gru']['dropout']),
                          bidirectional=config['simple_gru']['bidirectional'])

        self.output_all = config['penalize_all_steps']
        num_directions = 2 if config['simple_gru']['bidirectional'] else 1
        self.output_scores = nn.Linear(in_features=num_directions * int(config['simple_gru']['hidden_size']),
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


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, config):
        super(SimpleLSTM, self).__init__()
        self.num_directions = 2 if config['simple_lstm']['bidirectional'] else 1
        self.penalize_all_steps = config['penalize_all_steps']
        self.hidden_size = int(config['simple_lstm']['hidden_size'])
        self.out_size = len(config['labels_to_int'])

        self.embedding_layer = nn.Embedding(input_size, int(config['simple_lstm']['embedding_size']), padding_idx=0,
                                            sparse=True)
        self.rnn = nn.LSTM(int(config['simple_lstm']['embedding_size']), self.hidden_size,
                          int(config['simple_lstm']['num_layers']), dropout=abs(config['simple_lstm']['dropout']),
                          bidirectional=config['simple_lstm']['bidirectional'] )

        self.dense = nn.Linear(in_features=self.hidden_size,
                                       out_features=self.out_size)

        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def get_non_sparse_parameters(self):
        return list(set(self.parameters()) - set(self.embedding_layer.parameters()))

    def get_sparse_parameters(self):
        return self.embedding_layer.parameters()

    def forward(self, x, y=None):
        seq_len, batch_size = x.size()[0], x.size()[1]
        x = self.embedding_layer(x)
        x, (h_n, c_n) = self.rnn(x)

        x = x.view(seq_len, batch_size, self.num_directions, self.hidden_size)
        x = self.dense(x)

        if self.num_directions == 2:
            predicted_labels = (self.softmax(x[-1, :, 0, :]) + self.softmax(x[0, :, 1, :])).argmax(-1)
        else:
            predicted_labels = (self.softmax(x[-1, :, 0, :])).argmax(-1)

        if y is not None:
            if self.penalize_all_steps:
                x = x.view(-1, self.out_size)
                y = y.repeat_interleave(self.num_directions)
                y = y.repeat(seq_len)

            else:
                if self.num_directions == 2:
                    x = torch.cat((x[-1, :, 0, :], x[0, :, 1, :]), dim=0)

                y = y.repeat(self.num_directions)

            loss = self.loss_fn(x, y)

            return predicted_labels, loss

        else:
            return predicted_labels
