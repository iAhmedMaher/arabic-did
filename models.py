import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig, BertModel
import torch
from salesforce.model import RNNModel
from salesforce.utils import repackage_hidden


def get_model(input_size, config):
    if config['model'] == 'simple_gru':
        return SimpleGRU(input_size, config)
    elif config['model'] == 'simple_lstm':
        return SimpleLSTM(input_size, config)
    elif config['model'] == 'bert_pretrained':
        return BertPreTrained(input_size, config)
    elif config['model'] == 'awd_rnn':
        return AWDRNN(input_size, config)
    elif config['model'] == 'vdcnn':
        return VDCNN(input_size, config)
    elif config['model'] == 'bert':
        return Bert(input_size, config)

    raise NotImplementedError()


class BertPreTrained(nn.Module):
    def __init__(self, input_size, config):
        super(BertPreTrained, self).__init__()
        self.transformer = BertForSequenceClassification.from_pretrained(config['bert_pretrained']['model'],
                                                                         num_labels=len(config['labels_to_int']))

    def forward(self, x, y=None):
        x = x.permute(1, 0)
        attn_mask = (x != 0).float()

        if y is not None:
            loss, logits = self.transformer(x, labels=y, attention_mask=attn_mask)
            predicted_labels = logits.argmax(-1)

            return predicted_labels, loss
        else:
            logits = self.transformer(x, attention_mask=attn_mask)
            return logits.argmax(-1)


    def get_non_sparse_parameters(self):
        return self.parameters()

    def get_sparse_parameters(self):
        return []


# TODO deprecated (won't work)
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
                           bidirectional=config['simple_lstm']['bidirectional'])

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
                else:
                    x = x[-1, :, 0, :]

                y = y.repeat(self.num_directions)

            loss = self.loss_fn(x, y)

            return predicted_labels, loss

        else:
            return predicted_labels


class Bert(nn.Module):
    def __init__(self, input_size, config):
        super(Bert, self).__init__()
        self.device = config['device']
        self.penalize_all_steps = config['penalize_all_steps']
        self.cls_token = input_size
        hidden_size = (int(config['bert']['hidden_size']) // int(config['bert']['n_att_heads'])) * int(config['bert']['n_att_heads'])
        self.out_size = len(config['labels_to_int'])

        bert_config = BertConfig(vocab_size_or_config_json_file=input_size+1,
                                 hidden_size=int(hidden_size),
                                 hidden_act='relu', max_position_embeddings=config['preprocessing']['max_seq_len'],
                                 type_vocab_size=1, num_hidden_layers=int(config['bert']['n_bert_layers']),
                                 num_attention_heads=int(config['bert']['n_att_heads']),
                                 intermediate_size=int(config['bert']['intermediate_dense_size']),
                                 hidden_dropout_prob=abs(config['bert']['hidden_dropout']),
                                 attention_probs_dropout_prob=abs(config['bert']['att_dropout']))

        self.bert = BertModel(bert_config)
        self.dense = nn.Linear(int(hidden_size), self.out_size)

        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y=None):
        seq_len, batch_size = x.size()[0], x.size()[1]
        # Add CLS token to sequence
        torch.cat([(torch.ones((1, batch_size), dtype=torch.long) * self.cls_token).to(self.device), x], dim=0)

        x = x.permute(1, 0)
        x = self.bert(x)[0]  # TODO supply your own embedding to avoid token_type embedding
        x = x.permute(1, 0, 2)

        x = self.dense(x)

        predicted_labels = (self.softmax(x[0, :, :])).argmax(-1)

        if y is not None:
            if self.penalize_all_steps:
                x = x.view(-1, self.out_size)
                y = y.repeat(seq_len)

            else:
                x = x[0, :, :]

            loss = self.loss_fn(x, y)

            return predicted_labels, loss

        else:
            return predicted_labels


    def get_non_sparse_parameters(self):
        return self.parameters()

    def get_sparse_parameters(self):
        return []


class AWDRNN(nn.Module):
    def __init__(self, input_size, config):
        super(AWDRNN, self).__init__()
        self.penalize_all_steps = config['penalize_all_steps']
        self.hidden_size = int(config['awd_rnn']['hidden_size'])
        self.out_size = len(config['labels_to_int'])
        self.ar_alpha = config['awd_rnn']['ar_alpha']

        self.penalize_all_steps = config['penalize_all_steps']
        self.rnn = RNNModel(rnn_type='LSTM', ntoken=input_size, ninp=int(config['awd_rnn']['embedding_size']),
                            nlayers=int(config['awd_rnn']['num_layers']),
                            nhid=int(config['awd_rnn']['hidden_size']), dropout=abs(config['awd_rnn']['dropouto']),
                            dropouth=abs(config['awd_rnn']['dropouth']), dropouti=abs(config['awd_rnn']['dropouti']),
                            dropoute=abs(config['awd_rnn']['dropoute']), wdrop=abs(config['awd_rnn']['wdrop']))

        self.dense = nn.Linear(in_features=int(config['awd_rnn']['hidden_size']),
                               out_features=len(config['labels_to_int']))

        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        seq_len, batch_size = x.size()[0], x.size()[1]

        hidden = self.rnn.init_hidden(batch_size)
        hidden = repackage_hidden(hidden)

        x, hidden, rnn_hs, dropped_rnn_hs = self.rnn(x, hidden, return_h=True)
        x = self.dense(x)

        predicted_labels = (self.softmax(x[-1, :, :])).argmax(-1)

        if y is not None:
            if self.penalize_all_steps:
                x = x.view(-1, self.out_size)
                y = y.repeat(seq_len)

            else:
                x = x[-1, :, :]

            loss = self.loss_fn(x, y) + sum(
                self.ar_alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])

            return predicted_labels, loss

        else:
            return predicted_labels

    def get_non_sparse_parameters(self):
        return self.parameters()

    def get_sparse_parameters(self):
        return []


class VDCNN(nn.Module):

    def __init__(self, input_size, config):
        super(VDCNN, self).__init__()
        self.device = config['device']
        self.k = max(1, int(config['vdcnn']['k']))

        self.embedding_layer = nn.Embedding(input_size, int(config['vdcnn']['embedding_size']), padding_idx=0,
                                            sparse=True)

        self.conv0 = nn.Conv1d(in_channels=int(config['vdcnn']['embedding_size']),
                               out_channels=max(32, int(config['vdcnn']['conv0_nfmaps'])), kernel_size=3, padding=1)

        current_fmaps = max(32, int(config['vdcnn']['conv0_nfmaps']))

        self.conv_parts = []
        self.temp = []
        self.max_pool = nn.MaxPool1d(3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=abs(config['vdcnn']['dropout']))
        min_fmaps = 32
        for i in range(1, 5):
            n_fmaps = max(min_fmaps, int(config['vdcnn']['conv' + str(i) + '_nfmaps']))
            n_blocks = max(0, int(config['vdcnn']['conv' + str(i) + '_nblocks']))

            current_blocks = []
            for j in range(n_blocks):
                block = ConvBlock(current_fmaps, n_fmaps, abs(config['vdcnn']['dropout']),
                                  config['vdcnn']['apply_shortcut'])
                current_blocks.append(block)
                self.temp.append(block)
                current_fmaps = n_fmaps

            self.conv_parts.append(current_blocks)
            min_fmaps *= 2

        current_features = current_fmaps * self.k
        self.dense_layers = []
        self.relu = nn.ReLU()

        dense_nlayers = max(1, int(config['vdcnn']['dense_nlayers']))

        for i in range(dense_nlayers - 1):
            new_features = max(64, int(config['vdcnn']['dense_nfeatures']))
            layer = nn.Linear(current_features, new_features)
            self.dense_layers += [layer]
            self.temp.append(layer)
            current_features = new_features

        layer = nn.Linear(current_features, len(config['labels_to_int']))
        self.dense_layers += [layer]
        self.temp.append(layer)

        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CrossEntropyLoss()

        self.temp2 = nn.Sequential(*self.temp)

    def kmax_pooling(self, x, dim):
        if x.size()[dim] < self.k:
            pad_size = list(x.size())
            pad_size[dim] = self.k - x.size()[dim]
            return torch.cat([x, torch.zeros(pad_size, dtype=torch.float).to(self.device)], dim=dim)
        index = x.topk(self.k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def forward(self, x, y=None):
        batch_size = x.size()[1]
        x = self.embedding_layer(x)

        x = x.permute(1, 2, 0)
        x = self.conv0(x)
        x = self.relu(x)
        x = self.dropout(x)

        for part in self.conv_parts:
            for block in part:
                x = block(x)
                x = self.dropout(x)

            x = self.max_pool(x)

        x = self.kmax_pooling(x, 2)
        x = x.view(batch_size, -1)

        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = self.relu(x)

        x = self.dense_layers[-1](x)

        predicted_labels = self.softmax(x).argmax(-1)

        if y is not None:

            loss = self.loss_fn(x, y)

            return predicted_labels, loss

        else:
            return predicted_labels

    def get_non_sparse_parameters(self):
        return list(set(self.parameters()) - set(self.embedding_layer.parameters()))

    def get_sparse_parameters(self):
        return self.embedding_layer.parameters()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, apply_shortcut):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.apply_shortcut = apply_shortcut

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(num_features=out_channels)

        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(num_features=out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        if in_channels != out_channels and self.apply_shortcut:
            self.conv_shortcut = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
            self.norm_shortcut = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        # Assumes x is (batch, channels, timesteps)
        if self.apply_shortcut:
            if self.in_channels != self.out_channels:
                x_prev = self.norm_shortcut(self.conv_shortcut(x))
            else:
                x_prev = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        if self.apply_shortcut:
            x = (x + x_prev) / 2
        x = self.relu(x)
        x = self.dropout(x)

        return x
