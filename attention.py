import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention(args):

    score = generalAttentionScore(encoder_hidden_size=args['encoder_hidden_size'] * 2, decoder_hidden_size=args['decoder_hidden_size'])
    return Attention(score)

class Attention(torch.nn.Module):

    def __init__(self, attn_score):
        super(Attention, self).__init__()
        self.attn_score = attn_score

    def attn_weights(self, hidden, encoder_outputs):
        scores = self.attn_score(hidden, encoder_outputs)
        return F.softmax(scores, dim=1)

    def attn_context(self, attn_weights, encoder_outputs):

        weights = attn_weights.unsqueeze(2)  # (batch, seq_len) -> (batch, seq_len, 1)
        encoder_out = encoder_outputs.permute(1, 2, 0)  # (seq_len, batch, enc_h) -> (batch, enc_h, seq_len)
        context = torch.bmm(encoder_out, weights)  # (batch, enc_h, 1)
        return context.squeeze(2)

    def forward(self, t, hidden, encoder_outputs):
        attn_weights = self.attn_weights(hidden, encoder_outputs)
        return attn_weights, self.attn_context(attn_weights, encoder_outputs)

class generalAttentionScore(nn.Module):

    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(generalAttentionScore, self).__init__()
        
        data = torch.Tensor(decoder_hidden_size, encoder_hidden_size)
        stdev = 1. / math.sqrt(decoder_hidden_size)
        data.normal_(-stdev, stdev)
        self.W = nn.Parameter(data)

    def forward(self, hidden, encoder_outputs):
        hW = hidden.mm(self.W)  # (batch, enc_h)
        hW = hW.unsqueeze(1)  # (batch, enc_h) -> (batch, 1, enc_h)
        enc_out = encoder_outputs.permute(1, 2, 0)  # (seq_len, batch, enc_h) -> (batch, enc_h, seq_len)
        scores = torch.bmm(hW, enc_out)  # (batch, 1, seq_len) # batch matrix matrix product
        score = scores.squeeze(1)
        #print("SHAPE OF score = ", score.shape)
        return score