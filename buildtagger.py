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

# model
MAX_SENT_LENGTH = 150 # train max is 141
NUM_EPOCHS = 1 # default 30?
EMBEDDING_DIM = 10
HIDDEN_DIM = 50
BATCH_SIZE = 10
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.5

# preferences
USE_CPU = False # default False, can overwrite
PRINT_ROUND = max(int(round(round((30000/BATCH_SIZE)/20, 0)/5,0)*5),5)
print('Will print for iters in multiples of', PRINT_ROUND)
BUILDING_MODE = False
REVIEW_MODE = False
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

class LSTMTagger(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, word_to_idx, tag_to_idx, batch_size, dropout_rate, ):
    super(LSTMTagger, self).__init__()
    # init terms
    self.batch_size = batch_size
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout_rate
    self.word_to_idx = word_to_idx
    self.tag_to_idx = tag_to_idx
    self.vocab_size = len(word_to_idx)
    self.tagset_size = len(tag_to_idx)

    # # Matrix of transition parameters.  Entry i,j is the score of
    # # transitioning *to* i *from* j.
    # self.transitions = nn.Parameter(
    #     torch.randn(self.tagset_size, self.tagset_size)).to('cpu')

    # # These two statements enforce the constraint that we never transfer
    # # to the start tag and we never transfer from the stop tag
    # self.transitions.data[tag_to_idx[START_TAG], :] = -10000
    # self.transitions.data[:, tag_to_idx[STOP_TAG]] = -10000

    # layers
    self.word_embeddings = nn.Embedding(
        num_embeddings = self.vocab_size, 
        embedding_dim = self.embedding_dim
    )
    self.lstm = nn.LSTM(
      self.embedding_dim, 
      self.hidden_dim//2, 
      batch_first=False,
      num_layers=1, 
      bidirectional=True
      )
    self.hidden = self.init_hidden()
    self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
  
  def init_hidden(self):
    return (torch.randn(2, self.batch_size, self.hidden_dim//2),
            torch.randn(2, self.batch_size, self.hidden_dim//2)
            )

  # def argmax(self, vec):
  #     # return the argmax as a python int
  #   _, idx = torch.max(vec, 1)
  #   return idx.item()

  # def _viterbi_decode(self, feats, word_to_ix, tag_to_ix):
  #   backpointers = []

  #   # Initialize the viterbi variables in log space
  #   init_vvars = torch.full((1, self.tagset_size), -10000.).to('cpu')
  #   init_vvars[0][tag_to_ix[START_TAG]] = 0

  #   # forward_var at step i holds the viterbi variables for step i-1
  #   forward_var = init_vvars
  #   for feat in feats:
  #       bptrs_t = []  # holds the backpointers for this step
  #       viterbivars_t = []  # holds the viterbi variables for this step

  #       for next_tag in range(self.tagset_size):
  #           # next_tag_var[i] holds the viterbi variable for tag i at the
  #           # previous step, plus the score of transitioning
  #           # from tag i to next_tag.
  #           # We don't include the emission scores here because the max
  #           # does not depend on them (we add them in below)
  #           next_tag_var = forward_var.to('cpu') + self.transitions[next_tag].to('cpu')
  #           best_tag_id = self.argmax(next_tag_var)
  #           bptrs_t.append(best_tag_id)
  #           viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
  #       # Now add in the emission scores, and assign forward_var to the set
  #       # of viterbi variables we just computed
  #       forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
  #       backpointers.append(bptrs_t)

  #   # Transition to STOP_TAG
  #   print('forward_var.shape:', forward_var.shape)
  #   print('self.transitions[tag_to_ix[STOP_TAG]]:', self.transitions[tag_to_ix[STOP_TAG]].shape)
  #   terminal_var = forward_var + self.transitions[tag_to_ix[STOP_TAG]]
  #   best_tag_id = self.argmax(terminal_var)
  #   path_score = terminal_var[0][best_tag_id]

  #   # Follow the back pointers to decode the best path.
  #   best_path = [best_tag_id]
  #   for bptrs_t in reversed(backpointers):
  #       best_tag_id = bptrs_t[best_tag_id]
  #       best_path.append(best_tag_id)
  #   # Pop off the start tag (we dont want to return that to the caller)
  #   start = best_path.pop()
  #   assert start == tag_to_ix[START_TAG]  # Sanity check
  #   best_path.reverse()
    
  #   return path_score, best_path

  def forward(self, sentence):   
    self.hidden = self.init_hidden()

    if REVIEW_MODE:
      print('sentence:', sentence.shape)
    embeds = self.word_embeddings(sentence)
    
    # nn.LSTM expects input shape of (seq, batch, features) assuming batch_first=False
    if REVIEW_MODE:
      print('embeds:', embeds.shape)
      print('embeds.view', embeds.view(sentence.shape[1], self.batch_size, self.embedding_dim).shape)
    lstm_out, self.hidden = self.lstm(embeds.view(sentence.shape[1], self.batch_size, self.embedding_dim))
    #lstm_dropout = F.dropout(lstm_out, p=self.dropout_rate)
    
    # nn.Linear expects the input shape og (batch, *, features)
    if REVIEW_MODE:
      print('lstm_out:', lstm_out.shape)
      print('lstm_out.view', lstm_out.view(self.batch_size, sentence.shape[1], self.hidden_dim).shape)
    tag_space = self.hidden2tag(lstm_out.view(self.batch_size, sentence.shape[1], self.hidden_dim))
    
    if REVIEW_MODE:
      print('tag_space:', tag_space.shape)
      print(tag_space)
    #tag_scores, predicted = self._viterbi_decode(tag_space.to('cpu'), self.word_to_idx, self.tag_to_idx)
    tag_scores = F.log_softmax(tag_space, dim=1)
    predicted = torch.argmax(tag_scores.view(self.batch_size, -1, sentence.shape[1]), dim=1)

    # print('sentence:', sentence.device)
    # print('embeds:', embeds.device)
    # print('lstm_out:', lstm_out.device)
    # print('tag_space:', tag_space.device)
    # print('tag_scores:', tag_scores.device)
    # print('predicted:', predicted.device)

    return tag_scores, predicted


def train(model, training_generator, loss_function, optimizer, epoch):
  results = {'train_loss': [], 'train_acc': []}
  model.batch_size = BATCH_SIZE # update model with new batch_size
  # if epoch < NUM_EPOCHS/2 and BATCH_SIZE>50:
  #   model.batch_size = 50
  # else:
  #   model.batch_size = BATCH_SIZE # update model with new batch_size
  # print(model.batch_size)

  ##### TRAINING #####
  while datetime.datetime.now() - start_time < datetime.timedelta(minutes=MINS_TIME_OUT, seconds=30):
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
      tag_scores, predicted = model(sentence_in)
      # Step 4. Compute the accuracy, loss, gradients, and update the parameters
      if REVIEW_MODE:
        print('tag_scores:', tag_scores.shape)
        print('target_out:', target_out.shape)
      loss = loss_function(tag_scores.view(BATCH_SIZE, -1, MAX_SENT_LENGTH), target_out)
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

    tag_scores, predicted = model(sentence_in)
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
      #train_backup = data.copy()
      print('Building mode activated...')
      data = data[0:150].copy()

  # initialise dict
  char_to_idx = {'<PAD>': 0, UNKNOWN_TAG: 1}
  word_to_idx = {'<PAD>':0, UNKNOWN_TAG: 1}
  tag_to_idx = {'<PAD>':0, UNKNOWN_TAG: 1}

  # 0 is saved for padding
  char_idx = len(char_to_idx.keys())
  word_idx = len(word_to_idx.keys())
  tag_idx = len(tag_to_idx.keys())

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

def train_model(train_file, model_file):
  """ Main Training Pipeline

  Args:
      train_file (str): path to train file
      model_file (str): path to model file
  """
  torch.cuda.empty_cache()
  training_generator, validation_generator, word_to_idx, tag_to_idx = load_data_to_generators(train_file)
  model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, word_to_idx, tag_to_idx, BATCH_SIZE, DROPOUT_RATE).to(device)
  #model = BiLSTM_CRF(len(word_to_idx), tag_to_idx, EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE).to(device)
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
