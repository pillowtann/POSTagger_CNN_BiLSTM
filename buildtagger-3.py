# source: https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1%20-%20BiLSTM%20for%20PoS%20Tagging.ipynb
# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import re
import random
import numpy as np
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

##### CONSTANTS/SETTINGS #####
# reproducibility
start_time = datetime.datetime.now()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True

# data set-up
IGNORE_CASE = True
VAL_PROP = 0.2
GENERATOR_PARAMS = {
    'shuffle': True,
    'num_workers': 4,
    'drop_last': True, #ignore last incomplete batch
    'pin_memory': True
}
START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNKNOWN_TAG = "<UNK>"
PADDING_TAG = "<PAD>"

# model
MAX_SENT_LENGTH = 150 # train max is 141
NUM_EPOCHS = 1 # default 30?
EMBEDDING_DIM = 1024
HIDDEN_DIM = 1024
BATCH_SIZE = 10
LEARNING_RATE = 0.5
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
PAD_IDX = 0

# preferences
USE_CPU = False # default False, can overwrite
BUILDING_MODE = False
REVIEW_MODE = False
if BUILDING_MODE:
  TOTAL_DATA = 150
else:
  TOTAL_DATA = 30000
PRINT_ROUND = max(int(round(round((TOTAL_DATA/BATCH_SIZE)/20, 0)/5,0)*5),5)
print('Will print for iters in multiples of', PRINT_ROUND)
MINS_TIME_OUT = 2

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

def pad_sequence(array_seq, pad_value = 0):
  if MAX_SENT_LENGTH>len(array_seq):
      fixed_length_array = np.append(array_seq, np.zeros(MAX_SENT_LENGTH-len(array_seq)))
  else:
      fixed_length_array = array_seq[0:MAX_SENT_LENGTH-1]
  return fixed_length_array

def pad_tensor(tensor, pad_value = 0):
  if MAX_SENT_LENGTH>tensor.shape[0]:
      pads = torch.tensor(np.zeros(MAX_SENT_LENGTH-tensor.shape[0]), dtype=torch.long)
      fixed_length_tensor = torch.cat((tensor, pads), dim=0)
  else:
      fixed_length_tensor = tensor[0:MAX_SENT_LENGTH-1]
  return fixed_length_tensor

def calc_accuracy(predicted, target_out, sentence_in, batch_size):
  if batch_size>1:
    # calculates weighted accuracy over samples
    train_acc_num = 0
    train_acc_denom = 0
    for pred_row, act_row in zip(predicted.squeeze(), target_out.squeeze()):
        train_acc_num += sum([1 for pred, act in zip(pred_row, act_row) if (pred==act) and (pred!=0)])
        train_acc_denom += (sentence_in != 0).sum().item()
    train_acc = train_acc_num/train_acc_denom
  else:
    # calculates accuracy per batch sample
    train_acc = sum([1 for pred, act in zip(predicted.squeeze(), target_out.squeeze()) 
                      if (pred==act) and (pred!=0)])/(sentence_in != 0).sum().item()
  return train_acc

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

class BiLSTMPOSTagger(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
        
    def forward(self, text):

        #text = [sent len, batch size]
        
        #pass text through embedding layer
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        #pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        
        #outputs holds the backward and forward hidden states in the final layer
        #hidden and cell are the backward and forward hidden and cell states at the final time-step
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden/cell = [n layers * n directions, batch size, hid dim]
        
        #we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))
        
        #predictions = [sent len, batch size, output dim]
        
        return predictions


def train(model, training_generator, loss_function, optimizer, epoch):
  results = {'train_loss': [], 'train_acc': []}
  model.batch_size = BATCH_SIZE # update model with new batch_size

  ##### TRAINING #####
  for idx, (sentence_in, target_out) in enumerate(training_generator):
    if datetime.datetime.now() - start_time < datetime.timedelta(minutes=MINS_TIME_OUT, seconds=30):
      if device != torch.device("cpu"):
        # Move data to GPU if available
        sentence_in, target_out = sentence_in.to(device), target_out.to(device)
      if REVIEW_MODE:
        print(sentence_in.shape)
        print(target_out.shape)
      # Step 1. Remember that Pytorch accumulates gradients.
      # We need to clear them out before each instance
      model.zero_grad()
      # Step 3. Run our forward pass.
      tag_scores = model(sentence_in)
      # Step 4. Compute the accuracy, loss, gradients, and update the parameters
      if REVIEW_MODE:
        print('tag_scores:', tag_scores.shape)
        print('target_out:', target_out.shape)
      loss = loss_function(tag_scores.view(model.batch_size, model.output_dim, -1), target_out)
      predicted = torch.argmax(tag_scores.view(model.batch_size, model.output_dim, -1), dim=1).detach()
      train_acc = calc_accuracy(predicted, target_out, sentence_in, model.batch_size)
      loss.backward()
      optimizer.step()
      # Store Info
      results['train_loss'].append(loss.data.item())
      results['train_acc'].append(train_acc)
      
      if idx%PRINT_ROUND==0:
        print('Step {} | Training Loss: {}, Accuracy: {}'.format(idx, round(loss.data.item(),2), round(train_acc*100,2)))
    else:
      break
  return model, results

def validate(model, validation_generator):
  """Validation on Pytorch

  Args:
      model (torch.): [description]
      validation_generator (torch.utils.data.dataloader.DataLoader): [description]

  Returns:
      results (dict): stores validation results
  """
  results = {'val_acc': []}

  ##### VALIDATION #####
  for sentence_in, target_out in validation_generator:
    val_batch_size = len(target_out)
    model.batch_size = val_batch_size # update model with new batch_size 
    if device != torch.device("cpu"):
        # Move data to GPU if available
        sentence_in, target_out = sentence_in.to(device), target_out.to(device)
    # Step 3. Run our forward pass.
    tag_scores = model(sentence_in)
    # Step 4. Compute the accuracy, loss, gradients, and update the parameters
    if REVIEW_MODE:
        print('tag_scores:', tag_scores.shape)
        print('target_out:', target_out.shape)
    predicted = torch.argmax(tag_scores.view(model.batch_size, model.output_dim, -1), dim=1).detach()
    val_acc = calc_accuracy(predicted, target_out, sentence_in, val_batch_size)

    # Store Info
    results['val_acc'].append(val_acc)
    print('Validation Accuracy: {}'.format(round(val_acc*100,2)))
  
  return results

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

  for row in data:

    data_row = row.split(' ')
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

  if BUILDING_MODE:
    # Randomly sample
    idx = [i for i in range(len(tokenized_words))]
    random.seed(1)
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

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)

def train_model(train_file, model_file):
  """ Main Training Pipeline

  Args:
      train_file (str): path to train file
      model_file (str): path to model file
  """
  torch.cuda.empty_cache()
  training_generator, validation_generator, word_to_idx, tag_to_idx = load_data_to_generators(train_file)
  model = BiLSTMPOSTagger(len(word_to_idx), EMBEDDING_DIM, HIDDEN_DIM, len(tag_to_idx), N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX).to(device)
  model.apply(init_weights)
  pretrained_embeddings = torch.rand(len(word_to_idx), EMBEDDING_DIM)
  model.embedding.weight.data.copy_(pretrained_embeddings)
  model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
  loss_function = nn.CrossEntropyLoss().to(device)
  optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
  results_dict = {}
  for epoch in range(NUM_EPOCHS):
    print('Epoch {}'.format(epoch), '-'*80)
    model, results = train(model, training_generator, loss_function, optimizer, epoch)
    if validation_generator is not None:
      with torch.no_grad():
          results.update(validate(model, validation_generator)) # validate and update results
      results_dict[epoch] = results
  torch.save(model.state_dict(), 'model_dict')
  torch.save(model, model_file)
  print('Finished...')
		
if __name__ == "__main__":
  # make no changes here
  train_file = sys.argv[1]
  model_file = sys.argv[2]
  train_model(train_file, model_file)
