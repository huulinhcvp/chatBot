import torch
from encoder import encoder
from decoder import decoder
from collections import OrderedDict
import random
import string

def sample(encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_len):
    batch_size = encoder_outputs.size(1)
    sequences = None

    input_word = torch.tensor([sos_idx] * batch_size)
    kwargs = {}
    for t in range(max_len):
        output, attn_weights, kwargs = decoder(t, input_word, encoder_outputs, h_n, **kwargs)
        _, argmax = output.max(dim=1)  # greedily take the most probable word
        input_word = argmax
        argmax = argmax.unsqueeze(1)  # (batch) -> (batch, 1) because of concatenating to sequences
        sequences = argmax if sequences is None else torch.cat([sequences, argmax], dim=1)

    # ensure there is EOS token at the end of every sequence (important for calculating lengths)
    end = torch.tensor([eos_idx] * batch_size).unsqueeze(1)  # (batch, 1)
    sequences = torch.cat([sequences, end], dim=1)

    # calculate lengths
    _, lengths = (sequences == eos_idx).max(dim=1)

    return sequences, lengths


class SequenceToSequenceTrain(torch.nn.Module):
    def __init__(self, encoder, decoder, vocab_size):

        super(SequenceToSequenceTrain, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size

    def forward(self, question, answer):
        answer_seq_len = answer.size(0)
        outputs = None

        # encode question sequence
        encoder_outputs, h_n = self.encoder(question)

        kwargs = {}
        input_word = answer[0]  # sos for whole batch
        #print("INPUT WORD = ", input_word.shape)
        for t in range(answer_seq_len - 1):
            output, attn_weights, kwargs = self.decoder(t, input_word, encoder_outputs, h_n, **kwargs)

            out = output.unsqueeze(0)  # (batch_size, vocab_size) -> (1, batch_size, vocab_size)
            outputs = out if outputs is None else torch.cat([outputs, out], dim=0)

            teacher_forcing = random.random() < 0.5
            if teacher_forcing:
                input_word = answer[t + 1]  # +1 input word for next timestamp
            else:
                _, argmax = output.max(dim=1)
                input_word = argmax  # index of most probable token (for whole batch)

        return outputs


class SequenceToSequencePredict(torch.nn.Module):

    def __init__(self, encoder, decoder, field):
        super(SequenceToSequencePredict, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = field.vocab.stoi['<sos>']
        self.eos_idx = field.vocab.stoi['<eos>']
        self.field = field

    def decode_sequence(self, tokens_idx):
        sequence = ''; checker = 0
        for idx in tokens_idx:
            token = self.field.vocab.itos[idx]
            if token not in string.punctuation and token[0] not in ('\'', ' '):
                sequence += ' '
            if token == '<eos>':
                break
            if token not in ('<', '>', ' '):
                if token == 'url':
                    sequence += '<url>'
                elif token == 'dm':
                    sequence += 'direct message'
                else:
                    sequence += token
                #print("TOKKK =, ", tok)
        return sequence.strip()

    def forward(self, questions, sampler, max_len):
        # raw strings to tensor
        q = self.field.process([self.field.preprocess(question) for question in questions])

        # encode question sequence
        encoder_outputs, h_n = self.encoder(q)

        # sample output sequence
        sequences, lengths = sample(encoder_outputs, h_n, self.decoder, self.sos_idx, self.eos_idx, max_len)

        # torch tensors -> python lists
        batch_size = sequences.size(0)
        sequences, lengths = sequences.tolist(), lengths.tolist()

        # decode output (token idx -> token string)
        seqs = []
        for batch in range(batch_size):
            seq = sequences[batch][:lengths[batch]]
            seqs.append(self.decode_sequence(seq))

        return seqs


def training(args, metadata):
    en = encoder(args, metadata)
    de = decoder(args, metadata)
    return SequenceToSequenceTrain(en, de, metadata.vocab_size)


def predict_answers(args, metadata, model_path, field):
    train = training(args, metadata)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        key = key[7:]
        new_state_dict[key] = value
    train.load_state_dict(new_state_dict)
    return SequenceToSequencePredict(train.encoder, train.decoder, field)