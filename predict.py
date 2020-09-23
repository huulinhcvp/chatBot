import torch
import torch.nn as nn
import os
import argparse
from model import predict_answers
import pickle
from torchtext import data
import collections


class PredictModel(nn.Module):

    def __init__(self, model):
        super(PredictModel, self).__init__()
        self.model = model

    def forward(self, question, sampling_strategy, max_len):
        return self.model([question], sampling_strategy, max_len)[0]

def main():
    torch.set_grad_enabled(False)
    cuda = False
    if torch.cuda.is_available():
        cuda = True
    model_path = 'Models/' + 'seq2seq-' + '500'
    
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    device = torch.device('cuda' if cuda else 'cpu')
    print("DEVICE = ", device)
    with open('Models/args', 'rb') as f:
        model_parameters = pickle.load(f)
    with open('Models/vocab', 'rb') as f:
        vocab = pickle.load(f)
    print('Objects for predict_answers loaded.')

    field = data.Field(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', tokenize='spacy', lower=True)
    #print(type(field))
    field.vocab = vocab

    Metadata = collections.namedtuple('Metadata', 'vocab_size padding_idx vectors')
    metadata = Metadata(vocab_size=len(vocab), padding_idx=vocab.stoi['<pad>'], vectors=vocab.vectors)

    model = PredictModel(predict_answers(model_parameters, metadata, model_path, field))
    #print('KEYBOARD\'s GROUP PREDICT MODEL =', model)
    model.eval()

    question = ''
    print('\n\nMarketing staff: Hello!!! How can I help you?', flush=True)
    while question.lower() != 'bye':
        while True:
            print('Client: ', end='')
            question = input()
            if question:
                break

        answers = model(question, 'abc', 62)
        print('Marketing staff: ' + answers)


if __name__ == '__main__':
    main()
