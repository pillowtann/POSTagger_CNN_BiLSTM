# python3.5 build_tagger.py <train_file_absolute_path> <model_file_absolute_path>
# python build_tagger.py sents.train model.json

import os
import math
import sys
import datetime
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
startTime = datetime.datetime.now()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True).to(device)
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size).to(device)
        self.hidden = self.init_hidden(50)

    def init_hidden(self, seq_length):
        return (torch.zeros(2, seq_length, self.hidden_dim).to(device),
                torch.zeros(2, seq_length, self.hidden_dim).to(device))

    def forward(self, sentence):
        batch_size, seq_len = sentence.size()
        embeds = self.word_embeddings(sentence).to(device)
        embeds = embeds.contiguous().to(device)
        input_x = embeds.view(seq_len, -1, batch_size).to(device)
        _lstm_out, self.hidden = self.lstm(input_x, self.hidden)
        lstm_out = _lstm_out.contiguous().to(device)
        tag_space = self.hidden2tag(lstm_out.view(-1, self.hidden_dim*2)).to(device)
        tag_scores = F.log_softmax(tag_space, dim=1).to(device)

        # print('sentence.shape:', sentence.shape)
        # print('embeds.shape:', embeds.shape)
        # print('input_x.shape:', input_x.shape)
        # print('_lstm_out.shape:', _lstm_out.shape)
        # print('lstm_out.shape:', lstm_out.shape)
        # print('tag_space.shape:', tag_space.shape)
        # print('tag_scores.shape:', tag_scores.shape)

        return tag_scores.to(device)

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
		# use torch library to save model parameters, hyperparameters, etc. to model_file
    tFile = open(train_file, "r")
    training_data = []
    word_to_ix = {}
    word_to_ix["<PAD>"] = 0
    word_to_ix["<UNK>"] = 1
    tag_to_ix = {}
    tag_to_ix["<PAD>"] = 0

    for line in tFile:
        wordarr = []
        tagarr = []
        for Tword in line.split():
            word = Tword.rpartition("/")[0]
            tag = Tword.rpartition("/")[2]
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
            wordarr.append(word)
            tagarr.append(tag)
        training_data.append((wordarr, tagarr))
    start = 0
    stop = 32
    batches = []
    while (start < len(training_data)):
        if (stop > len(training_data)):
            batches.append(training_data[start: len(training_data)])
            for x in range(len(training_data), stop):
                batches[len(batches) - 1].append(([], []))
        else:
            batches.append(training_data[start: stop])
        start += 32
        stop += 32
    max_max_length = 0
    
    for batch in batches:
        max_length = max(len(sentence) for sentence, tags in batch)
        if max_length > max_max_length:
            max_max_length = max_length
        for sentence, tags in batch:
            for x in range(len(sentence), max_length):
                sentence.append("<PAD>")
                tags.append("<PAD>")

    model = LSTMTagger(32, max_max_length, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    print(word_to_ix)
    while datetime.datetime.now() - startTime < datetime.timedelta(minutes=9, seconds=30):
        for batch in batches:
            sents = []
            taglist = []
            model.zero_grad()
            max_length = 0
            for sentence, tags in batch:
                max_length = len(sentence)
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = prepare_sequence(tags, tag_to_ix)
                sents.append(sentence_in)
                taglist.append(targets)
            sents = torch.stack(sents, 0).to(device)
            model.hidden = model.init_hidden(max_length)
            taglist = torch.stack(taglist, 0).to(device)
            tag_scores = model(sents).to(device)
            taglist = taglist.view(-1).to(device)
            print('taglist.shape:', taglist.shape)
            loss = loss_function(tag_scores, taglist).to(device)
            loss.backward()
            optimizer.step()
            print('loss.data.item():', loss.data.item())
    data = {}
    data["model"] = model
    data["word_to_ix"] =  word_to_ix
    data["tag_to_ix"] = tag_to_ix
    with open(model_file, "wb") as outfile:
        pickle.dump(data, outfile)
    outfile.close()
    tFile.close()
    print("--- %s seconds ---" % (datetime.datetime.now() - startTime))
    print('Finished...')
		
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)