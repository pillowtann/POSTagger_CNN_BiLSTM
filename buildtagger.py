# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import re
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


##### CONSTANTS/SETTINGS #####
# reproducibility
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

# model
MAX_SENT_LENGTH = 150 # train max is 141
NUM_EPOCHS = 1 # default 30?
EMBEDDING_DIM = 10
HIDDEN_DIM = 50
BATCH_SIZE = 50
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.5

# preferences
USE_CPU = False # default False, can overwrite
PRINT_ROUND = max(int(round(round((30000/BATCH_SIZE)/20, 0)/5,0)*5),5)
print('Will print for iters in multiples of', PRINT_ROUND)
BUILDING_MODE = False
REVIEW_MODE = False

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
        fixed_length_tensor = array_seq[0:MAX_SENT_LENGTH-1]
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

class LSTMTagger(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, batch_size, dropout_rate):
      super(LSTMTagger, self).__init__()
      # layers
      self.word_embeddings = nn.Embedding(
          num_embeddings = vocab_size, 
          embedding_dim = embedding_dim
      )
      self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False, dropout_rate=DROPOUT_RATE)
      self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
      
      # init terms
      self.batch_size = batch_size
      self.embedding_dim = embedding_dim
      self.hidden_dim = hidden_dim
      self.dropout_rate = dropout_rate
  
  def forward(self, sentence):   
      
      if REVIEW_MODE:
          print('sentence:', sentence.shape)
      embeds = self.word_embeddings(sentence)
      
      # nn.LSTM expects input shape of (seq, batch, features) assuming batch_first=False
      if REVIEW_MODE:
          print('embeds:', embeds.shape)
          print('embeds.view', embeds.view(sentence.shape[1], self.batch_size, self.embedding_dim).shape)
      lstm_out, _ = self.lstm(embeds.view(sentence.shape[1], self.batch_size, self.embedding_dim))
      #lstm_dropout = F.dropout(lstm_out, p=self.dropout_rate)
      
      # nn.Linear expects the input shape og (batch, *, features)
      if REVIEW_MODE:
          print('lstm_out:', lstm_out.shape)
          print('lstm_out.view', lstm_out.view(self.batch_size, sentence.shape[1], self.hidden_dim).shape)
      tag_space = self.hidden2tag(lstm_out.view(self.batch_size, sentence.shape[1], self.hidden_dim))
      
      if REVIEW_MODE:
          print('tag_space:', tag_space.shape)
      tag_scores = F.log_softmax(tag_space, dim=1)
      
      return tag_scores

def train(model, training_generator, loss_function, optimizer, epoch):
    results = {'train_loss': [], 'train_acc': []}
    model.batch_size = BATCH_SIZE # update model with new batch_size
    # if epoch < NUM_EPOCHS/2 and BATCH_SIZE>50:
    #   model.batch_size = 50
    # else:
    #   model.batch_size = BATCH_SIZE # update model with new batch_size
    # print(model.batch_size)

    ##### TRAINING #####
    for idx, (sentence_in, target_out) in enumerate(training_generator):
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
        loss = loss_function(tag_scores.view(BATCH_SIZE, -1, MAX_SENT_LENGTH), target_out)
        predicted = torch.argmax(tag_scores.view(BATCH_SIZE, -1, MAX_SENT_LENGTH), dim=1)
        predicted = predicted.detach()
        train_acc = calc_accuracy(predicted, target_out, sentence_in, BATCH_SIZE)
        loss.backward()
        optimizer.step()
        # Store Info
        results['train_loss'].append(loss.data.item())
        results['train_acc'].append(train_acc)
        
        if idx%PRINT_ROUND==0:
            print('Step {} | Training Loss: {}, Accuracy: {}'.format(idx, round(loss.data.item(),2), round(train_acc*100,2)))
    
    return model, results

def validate(model, validation_generator, val_batch_size):
  """Validation on Pytorch

  Args:
      model (torch.): [description]
      validation_generator (torch.utils.data.dataloader.DataLoader): [description]
      val_batch_size (int): batch size for validation set (should be full length)

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

    tag_scores = model(sentence_in)
    predicted = torch.argmax(tag_scores.view(val_batch_size, -1,MAX_SENT_LENGTH), dim=1)
    predicted = predicted.detach()
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
      train_backup = data.copy()
      data = train[0:150].copy()

  # initialise dict
  char_to_idx = {'<PAD>': 0}
  word_to_idx = {'<PAD>':0}
  tag_to_idx = {'<PAD>':0}
  # 0 is saved for padding
  char_idx = 1
  word_idx = 1
  tag_idx = 1

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
          set(list(word))
          
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
              char_to_idx[ch] = char_idx
              char_idx += 1
          
          # add to word dict if new
          if word not in word_to_idx.keys():
              word_to_idx[word] = word_idx
              word_idx += 1

          # add to tag dict if new
          if tag not in tag_to_idx.keys():
              tag_to_idx[tag] = tag_idx
              tag_idx += 1

          # add tokenized to sequence
          _tokn_row_words.append(word) # do not tokenize for character embeddings later
          _tokn_row_tags.append(tag)

      tokenized_words.append(np.array(_tokn_row_words))
      tokenized_tags.append(np.array(_tokn_row_tags))

  tokenized_words = np.array(tokenized_words)
  tokenized_tags = np.array(tokenized_tags)

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

def train_model(train_file, model_file):
  """ Main Training Pipeline

  Args:
      train_file (str): path to train file
      model_file (str): path to model file
  """
  torch.cuda.empty_cache()
  training_generator, validation_generator, word_to_idx, tag_to_idx = load_data_to_generators(train_file)
  model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_idx), len(tag_to_idx), BATCH_SIZE, DROPOUT_RATE).to(device)
  loss_function = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  results_dict = {}
  for epoch in range(NUM_EPOCHS):
      print('Epoch {}'.format(epoch), '-'*80)
      model, results = train(model, training_generator, loss_function, optimizer, epoch)
      with torch.no_grad():
          results.update(validate(model, validation_generator)) # validate and update results
      results_dict[epoch] = results
  torch.save(model.state_dict(), model_file)
  print('Finished...')
		
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
