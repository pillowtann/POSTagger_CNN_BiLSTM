# python3.5 build_tagger.py <train_file_absolute_path> <model_file_absolute_path>
# python build_tagger.py sents.train model.json

import os
import math
import sys

import datetime
import pickle
import re
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

startTime = datetime.datetime.now()

##### CONSTANTS/SETTINGS #####
# reproducibility
start_time = datetime.datetime.now()
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic = True

# data set-up
IGNORE_CASE = True
VAL_PROP = 0
GENERATOR_PARAMS = {
    'shuffle': True,
    'num_workers': 1,
    'drop_last': True, #ignore last incomplete batch
    'pin_memory': True
}
START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNKNOWN_TAG = "<UNK>"
PADDING_TAG = "<PAD>"
PAD_IDX = 0

# model
MAX_SENT_LENGTH = 150 # train max is 141
NUM_EPOCHS = 30
EMBEDDING_DIM = 100
HIDDEN_DIM = 50
BATCH_SIZE = 1
DROPOUT_RATE = 0.2
MOMENTUM = 0.1
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.5 #1e-1

# preferences
USE_CPU = False # default False, can overwrite
BUILDING_MODE = False
REVIEW_MODE = False
TIME_OUT = False # do not change
if BUILDING_MODE:
  TOTAL_DATA = 20
else:
  TOTAL_DATA = 30000
PRINT_ROUND = max(int(round(round((TOTAL_DATA/BATCH_SIZE)/20, 0)/5,0)*5),1)
print('Will print for iters in multiples of', PRINT_ROUND)
MINS_TIME_OUT = 9

# gpu
if torch.cuda.is_available() and not USE_CPU:
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def pad_tensor(tensor, pad_value = 0):
  if MAX_SENT_LENGTH>tensor.shape[0]:
    pads = torch.tensor(np.zeros(MAX_SENT_LENGTH-tensor.shape[0]), dtype=torch.long)
    fixed_length_tensor = torch.cat((tensor, pads), dim=0)
  else:
    fixed_length_tensor = tensor[0:MAX_SENT_LENGTH-1]
  return fixed_length_tensor

class FormatDataset(Dataset):
  def __init__(self, processed_in, processed_out, word_to_idx, tag_to_idx):
    self.processed_in = processed_in
    self.processed_out = processed_out
    self.word_to_idx = word_to_idx
    self.tag_to_idx = tag_to_idx
          
  def __len__(self):
    # Run multiple rows at once, i.e. reduce enumerate
    return len(self.processed_in)

  def __getitem__(self, index):
    # Step 2. Get our inputs ready for the network, that is, turn them into
    # Tensors of word indices.
    sentence_in = pad_tensor(prepare_sequence(self.processed_in[index], self.word_to_idx))
    target_out = pad_tensor(prepare_sequence(self.processed_out[index], self.tag_to_idx))
    
    return sentence_in, target_out

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, pad_idx, dropout_rate):
        super(LSTMTagger, self).__init__()
        # values
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size # not used
        self.tagset_size = tagset_size # not used
        self.dropout_rate = dropout_rate # not used
        self.pad_idx = pad_idx

        # layers
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden = self.init_hidden(1) # initialising, number does not matter

    def init_hidden(self, seq_length):
        return (torch.zeros(2, seq_length, self.hidden_dim//2).to(device),
                torch.zeros(2, seq_length, self.hidden_dim//2).to(device))

    def forward(self, sentence, orig_seq_lengths):

        # print('sentence.shape:', sentence.shape)
        batch_size, seq_len = sentence.size()
        embeds = self.word_embeddings(sentence)
        # print('embeds.shape:', embeds.shape)
        embeds = embeds.contiguous()
        # print('embeds.shape:', embeds.shape)
        # input_x = embeds.view(seq_len, -1, batch_size)
        # print('input_x.shape:', input_x.shape)
        packed_input = pack_padded_sequence(embeds, orig_seq_lengths, batch_first=True)
        packed_output, self.hidden = self.lstm(packed_input, self.hidden)
        lstm_out, _ = pad_packed_sequence(packed_output, total_length=seq_len, batch_first=True)
        # print('lstm_out.shape:', lstm_out.shape)
        lstm_dropout = self.dropout(lstm_out.contiguous())
        # print('lstm_dropout.shape:', lstm_dropout.shape)
        tag_space = self.hidden2tag(lstm_dropout)
        # print('tag_space.shape:', tag_space.shape)
        tag_scores = F.log_softmax(tag_space, dim=2)
        # print('tag_scores.shape:', tag_scores.shape)
        tag_scores = tag_scores.contiguous().view(batch_size, -1, seq_len)
        # print('tag_scores.shape:', tag_scores.shape)

        return tag_scores

def load_data_to_generators(train_file):
  """Load data from specified path into train and validate data loaders

  Args:
      train_file (str): [description]

  Returns:
      training_generator:
      validation_generator:
      word_to_idx:
      tag_to_idx: 
  """
  ##### load train data #####
  print('Loading data...')
  with open(train_file, 'r') as fp:
    data = fp.readlines()
  print('Loaded train with', len(data), 'samples...')

  ### TRIAL RUN REDUCE DATASET ###
  if BUILDING_MODE:
    print('Building mode activated...')
    data = data[0:TOTAL_DATA].copy()

  # initialise dict
  char_to_idx = {PADDING_TAG: PAD_IDX, UNKNOWN_TAG: 1}
  word_to_idx = {PADDING_TAG: PAD_IDX, UNKNOWN_TAG: 1}
  tag_to_idx = {PADDING_TAG: PAD_IDX}

  # initialise processed lists
  tokenized_words = [] # model input
  tokenized_tags = [] # model output
  length_of_sequences = []

  for row in data:

    data_row = row.split(' ')
    length_of_sequences.append(len(data_row))

    _tokn_row_words = []
    _tokn_row_tags = []
        
    for wordtag in data_row:
      # split by '/' delimiter from the back once
      # e.g. Invest/Net/NNP causing problems with regular split
      word, tag = wordtag.rsplit('/', 1)

      ##### rules to format words #####
      # remove trailing newline \n
      word = word.rstrip()
      tag = tag.rstrip()
      
      # optional lowercase
      if IGNORE_CASE:
          word = word.lower() 
      
      # if word is numeric // [do blind] + tag is CD
      if (re.sub(r'[^\w]', '', word).isdigit()):
          word = '<NUM>'
          chars = ['<NUM>']
      else:
          chars = set(list(word))

      ##### store #####
      # add to char dict if new
      for ch in [c for c in chars if c not in char_to_idx.keys()]:
        char_to_idx[ch] = len(char_to_idx)
      
      # add to word dict if new
      if word not in word_to_idx.keys():
        word_to_idx[word] = len(word_to_idx)

      # add to tag dict if new
      if tag not in tag_to_idx.keys():
        tag_to_idx[tag] = len(tag_to_idx)

      # add tokenized to sequence
      _tokn_row_words.append(word) # do not tokenize for character embeddings later
      _tokn_row_tags.append(tag)

    tokenized_words.append(np.array(_tokn_row_words))
    tokenized_tags.append(np.array(_tokn_row_tags))

  tokenized_words = np.array(tokenized_words)
  tokenized_tags = np.array(tokenized_tags)

  if VAL_PROP>0:
    # Randomly sample
    idx = [i for i in range(len(tokenized_words))]
    random.seed(1234)
    random.shuffle(idx)
    split_idx = round(len(data)*VAL_PROP)
    _train_idx = idx[split_idx:]
    _val_idx = idx[0:split_idx]

    # Generators
    training_set = FormatDataset(tokenized_words[_train_idx], tokenized_tags[_train_idx], word_to_idx, tag_to_idx)
    training_generator = DataLoader(training_set, batch_size=BATCH_SIZE, **GENERATOR_PARAMS)

    val_batch_size = len(_val_idx)
    validation_set = FormatDataset(tokenized_words[_val_idx], tokenized_tags[_val_idx], word_to_idx, tag_to_idx)
    validation_generator = DataLoader(validation_set, batch_size=val_batch_size, **GENERATOR_PARAMS)

    return training_generator, validation_generator, word_to_idx, tag_to_idx
  
  else:
    # Generators
    training_set = FormatDataset(tokenized_words, tokenized_tags, word_to_idx, tag_to_idx)
    training_generator = DataLoader(training_set, batch_size=BATCH_SIZE, **GENERATOR_PARAMS)

    return training_generator, None, word_to_idx, tag_to_idx

def calc_accuracy(predicted, target_out, batch_size):
  if batch_size>1:
    # calculates weighted accuracy over samples
    train_acc_num = 0
    train_acc_denom = 0
    for pred_row, act_row in zip(predicted.squeeze(), target_out.squeeze()):
      train_acc_num += sum([1 for pred, act in zip(pred_row, act_row) if (pred==act) and (act!=0)])
      train_acc_denom += sum([1 for i in act_row if i!=0])
    train_acc = train_acc_num/train_acc_denom
  else:
    # calculates accuracy per batch sample
    train_acc = sum([1 for pred, act in zip(predicted[0], target_out[0]) 
                      if (pred==act) and (act!=0)])/sum([1 for i in target_out[0] if i!=0])
  return train_acc

def train_model(train_file, model_file):

    # load data
    training_generator, validation_generator, word_to_ix, tag_to_ix = load_data_to_generators(train_file)
    
    # load model
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), PAD_IDX, DROPOUT_RATE).to(device)

    # define loss and optimizer
    loss_function = nn.CrossEntropyLoss(ignore_index = PAD_IDX).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # train step
    results_dict = {}
    for epoch in range(NUM_EPOCHS):
        print('Epoch {}'.format(epoch), '-'*80)
        results = {'train_loss': [], 'train_acc': []}
        
        for idx, (sentence_in, target_out) in enumerate(training_generator):

            if datetime.datetime.now() - start_time > datetime.timedelta(minutes=MINS_TIME_OUT, seconds=30):
                TIME_OUT = True
                break

            # format batch data
            sentence_lengths = []
            for row in sentence_in:
                sentence_lengths.append(torch.nonzero(row).shape[0])
            sentence_lengths = torch.LongTensor(sentence_lengths)
            max_seq_len = max(sentence_lengths)
            seq_lengths, perm_idx = sentence_lengths.sort(0, descending=True)

            seq_lengths = seq_lengths.to(device)
            sentence_in = torch.narrow(sentence_in[perm_idx], dim=1, start=0, length=max_seq_len).to(device)
            target_out = torch.narrow(target_out[perm_idx], dim=1, start=0, length=max_seq_len).to(device)

            model.zero_grad()

            model.hidden = model.init_hidden(BATCH_SIZE)
            tag_scores = model(sentence_in, seq_lengths)
            predicted = torch.argmax(tag_scores, dim=1).detach().cpu().numpy()

            accuracy = calc_accuracy(predicted, target_out, BATCH_SIZE)
            loss = loss_function(tag_scores, target_out)
            loss.backward()
            optimizer.step()

            results['train_loss'].append(loss.data.item())
            results['train_acc'].append(accuracy)

            if idx%PRINT_ROUND==0:
                print('Step {} | Training Loss: {}, Accuracy: {}%'.format(idx, round(loss.data.item(),3), round(accuracy*100,3)))

        if BUILDING_MODE and epoch%5==0:
            print(predicted[0])
            print(target_out[0])

        results_dict[epoch] = results

        if TIME_OUT:
            break

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'word_to_ix': deepcopy(word_to_ix),
        'tag_to_ix': deepcopy(tag_to_ix),
        'embedding_dim': deepcopy(model.embedding_dim),
        'batch_size' : deepcopy(BATCH_SIZE),
        'hidden_dim': deepcopy(model.hidden_dim),
        'dropout_rate': deepcopy(model.dropout_rate),
        'pad_idx': deepcopy(model.pad_idx),
        'ignore_case': IGNORE_CASE,
        }, model_file)
    
    print('Time Taken: {}'.format(datetime.datetime.now() - startTime), '-'*60)
    print('Finished...')
		
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)