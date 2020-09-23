import torch
from abc import ABC, abstractmethod

def embeddings(args, metadata):
    embed = torch.nn.Embedding(num_embeddings=metadata.vocab_size, embedding_dim=args['embedding_size'], padding_idx=metadata.padding_idx, _weight=metadata.vectors)

    embed.weight.require_grads = False
    return embed

def encoder(args, metadata):
    embed = embeddings(args, metadata)
    return Encoder(
        rnn=getattr(torch.nn, 'LSTM'),
        embed=embed,
        embed_size=args['embedding_size'],
        hidden_size=args['encoder_hidden_size'],
        num_layers=args['encoder_num_layers'],
        dropout=args['encoder_rnn_dropout'],
        bidirectional=True
    )

class RNN(torch.nn.Module):

    def __init__(self, rnn):
        super(RNN, self).__init__()
        self.rnn = rnn

    def forward(self, *input):
        rnn_out, hidden = self.rnn(*input)
        hidden, s = hidden # bo qua LSTM cell state
        return rnn_out, hidden

class Encoder(torch.nn.Module):

    def __init__(self, rnn, embed, embed_size, hidden_size, num_layers=1, dropout=0.2,
                 bidirectional=False):
        super(Encoder, self).__init__()

        self._hidden_size = hidden_size
        self._bidirectional = bidirectional
        self._num_layers = num_layers

        self.embed = embed
        self.rnn = RNN(rnn(input_size=embed_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,bidirectional=bidirectional))

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def bidirectional(self):
        return self._bidirectional

    @property
    def num_layers(self):
        return self._num_layers

    def forward(self, input, h_0=None):
        word_embedded = self.embed(input)
        outputs, h_n = self.rnn(word_embedded, h_0)
        return outputs, h_n