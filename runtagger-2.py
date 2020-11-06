# python3.5 run_tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>
# python run_tagger.py sents.test model.json sents.out
# python eval.py sents.out sents.answer

import os
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True).to(device)
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size).to(device)
        self.hidden = self.init_hidden(50)

    def init_hidden(self, seq_length):
        return (torch.zeros(2, seq_length, self.hidden_dim).to(device),
                torch.zeros(2, seq_length, self.hidden_dim).to(device))

    def forward(self, sentence):
        seq_len = len(sentence)
        embeds = self.word_embeddings(sentence).to(device)
        input = embeds.view(seq_len, 1, -1).to(device)
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(seq_len, -1)).to(device)
        tag_scores = F.log_softmax(tag_space, dim=1).to(device)
        return tag_scores.to(device)

def tag_sentence(test_file, model_file, out_file):
    with open(model_file, "rb") as datafile:
        data = pickle.load(datafile)
    model = data["model"]
    word_to_ix = data["word_to_ix"] 
    tag_to_ix = data["tag_to_ix"]
    ix_to_tag = {}
    for tag in tag_to_ix:
        ix_to_tag[tag_to_ix[tag]] = tag
    tfile = open(test_file, "r")
    oFile = open(out_file, "w")
    for lines in tfile:
        line_arr = lines.split()
        line = []
        for word in line_arr:
            if word in word_to_ix:
                line.append(word)
            else:
                line.append("<UNK>")
        input = prepare_sequence(line, word_to_ix).to(device)
        model.hidden = model.init_hidden(len(line))
        tag_scores = model(input).to(device)
        sentence = []
        x = 0
        for tag_score in tag_scores:
            max, ix = tag_score.max(0)
            tag = ix_to_tag[ix.item()]
            if tag == "<PAD>":
                ix = tag_score.sort(descending=True)[1][1]
                tag = ix_to_tag[ix.item()]
            sentence.append(line[x] + "/" + tag)
            x += 1
        oFile.write(" ".join(sentence) + "\n")

    # write your code here. You can add functions as well.
		# use torch library to load model_file
    print('Finished...')

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)