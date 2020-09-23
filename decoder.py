import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from attention import attention

def embeddings(args, metadata):
    embed = nn.Embedding(num_embeddings=metadata.vocab_size, embedding_dim=args['embedding_size'],padding_idx=metadata.padding_idx, _weight=metadata.vectors)
    embed.weight.require_grads = False
    return embed


class DecoderInit(torch.nn.Module):
    def __init__(self, encoder_hidden_size, decoder_num_layers, decoder_hidden_size, rnn_cell_type):
        super(DecoderInit, self).__init__()
        self.linear = nn.Linear(in_features=encoder_hidden_size, out_features=decoder_hidden_size)
        self.decoder_num_layers = decoder_num_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.rnn_cell_type = rnn_cell_type #LSTM

    def forward(self, h_n):
        batch_size = h_n.size(1)
        backward_encoder = h_n[torch.arange(1, h_n.size(0), 2)]  # backward cua trang thai an cuoi cung cua ENCODER
        hidden = torch.tanh(self.linear(backward_encoder))
        return (hidden, torch.zeros(self.decoder_num_layers, batch_size,self.decoder_hidden_size))

def decoder(args, metadata):

    embed = embeddings(args, metadata)
    attn = attention(args)
    init = DecoderInit(args['encoder_hidden_size'], args['decoder_num_layers'], args['decoder_hidden_size'],'LSTM')
    return Decoder(
        rnn=getattr(nn, 'LSTM'),
        embed=embed,
        attn=attn,
        init_hidden=init,
        vocab_size=metadata.vocab_size,
        embed_size=args['embedding_size'],
        rnn_hidden_size=args['decoder_hidden_size'],
        attn_hidden_projection_size=args['attn_hidden_size'],
        encoder_hidden_size=args['encoder_hidden_size'] * 2,
        num_layers=args['decoder_num_layers'],
        dropout=args['decoder_rnn_dropout'],
        input_feed=True
    )

class DecoderDecorator(ABC, nn.Module):

    def __init__(self, *args):
        super(DecoderDecorator, self).__init__()
        self._args = []
        self._args_init = {}

    def forward(self, t, input, encoder_outputs, h_n, **kwargs):

        assert (t == 0 and not kwargs) or (t > 0 and kwargs)

        extra_args = []
        for arg in self.args:
            if t > 0 and arg not in kwargs:
                raise AttributeError("Mandatory arg \"%s\" not present among method arguments" % arg)
            extra_args.append(self.args_init[arg](encoder_outputs, h_n) if t == 0 else kwargs[arg])

        output, attn_weights, *out = self._forward(t, input, encoder_outputs, *extra_args)
        return output, attn_weights, {k: v for k, v in zip(self.args, out)}

    @abstractmethod
    def _forward(self, *args):
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_size(self):
        raise AttributeError

    @property
    @abstractmethod
    def num_layers(self):
        raise AttributeError

    @property
    @abstractmethod
    def has_attention(self):
        raise AttributeError

    @property
    def args(self):
        # "danh sach cac doi so bo sung ma lop con muon nhan"
        return self._args

    @args.setter
    def args(self, value):
        self._args = value

    @property
    def args_init(self):
        return self._args_init

    @args_init.setter
    def args_init(self, value):
        self._args_init = value


class Decoder(DecoderDecorator):

    LAST_STATE = 'last_state'
    LAST_ATTN_HIDDEN = 'last_attn_hidden'

    args = [LAST_STATE]

    def __init__(self, rnn, embed, attn, init_hidden, vocab_size, embed_size, rnn_hidden_size,
                 attn_hidden_projection_size, encoder_hidden_size, num_layers=1, dropout=0.2, input_feed=False):
        super(Decoder, self).__init__()

        if input_feed:
            self.args += [self.LAST_ATTN_HIDDEN]

        self.args_init = {
            self.LAST_STATE: lambda encoder_outputs, h_n: self.initial_hidden(h_n),
            self.LAST_ATTN_HIDDEN: lambda encoder_outputs, h_n: self.last_attn_hidden_init(h_n.size(1))  # h_n.size(1) == batch_size
        }

        self._hidden_size = rnn_hidden_size
        self._num_layers = num_layers
        self.initial_hidden = init_hidden

        self.input_feed = input_feed
        self.attn_hidden_projection_size = attn_hidden_projection_size

        rnn_input_size = embed_size + attn_hidden_projection_size
        self.embed = embed
        self.rnn = rnn(input_size=rnn_input_size,
                           hidden_size=rnn_hidden_size,
                           num_layers=num_layers,
                           dropout=dropout)
        self.attn = attn
        self.attn_hidden_lin = nn.Linear(in_features=rnn_hidden_size + encoder_hidden_size,
                                         out_features=attn_hidden_projection_size)
        self.out = nn.Linear(in_features=attn_hidden_projection_size, out_features=vocab_size)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def has_attention(self):
        return True

    def last_attn_hidden_init(self, batch_size):
        return torch.zeros(batch_size, self.attn_hidden_projection_size)

    def _forward(self, t, input, encoder_outputs, last_state, last_attn_hidden=None):
        assert (self.input_feed and last_attn_hidden is not None) or (not self.input_feed and last_attn_hidden is None)

        embedded = self.embed(input)

        # prepare rnn input
        rnn_input = embedded
        rnn_input = torch.cat([rnn_input, last_attn_hidden], dim=1) # inphut feeding (Paper 2)
        rnn_input = rnn_input.unsqueeze(0)  # (batch, rnn_input_size) -> (1, batch, rnn_input_size)

        # rnn output
        _, state = self.rnn(rnn_input, last_state)
        #print("HIDDEN STATE = ", state)

        hidden = state[0]
        #print("HIDEEN SHAPE = ", hidden.shape)

        # attention context
        attn_weights, context = self.attn(t, hidden[-1], encoder_outputs)
        #print("SHAPE OF CONtEXT = ", context.shape)
        attn_hidden = torch.tanh(self.attn_hidden_lin(torch.cat([context, hidden[-1]], dim=1)))  # (batch, attn_hidden)

        # calculate logits
        output = self.out(attn_hidden)
        #output = torch.nn.functional.softmax(output, dim = 1)

        return output, attn_weights, state, attn_hidden