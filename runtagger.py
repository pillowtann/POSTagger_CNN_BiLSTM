# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>


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

START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNKNOWN_TAG = "<UNK>"
MAX_SENT_LENGTH = 150

# preferences
USE_CPU = False # default False, can overwrite
BUILDING_MODE = False
REVIEW_MODE = False

# gpu
if torch.cuda.is_available() and not USE_CPU:
  device = torch.device("cuda:0")
  print("Running on the GPU")
else:
  device = torch.device("cpu")
  print("Running on the CPU")

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

    return tag_scores, predicted


def pad_tensor(tensor, pad_value = 0):
  if MAX_SENT_LENGTH>tensor.shape[0]:
      pads = torch.tensor(np.zeros(MAX_SENT_LENGTH-tensor.shape[0]), dtype=torch.long)
      fixed_length_tensor = torch.cat((tensor, pads), dim=0)
  else:
      fixed_length_tensor = tensor[0:MAX_SENT_LENGTH-1]
  return fixed_length_tensor

def pad_sequence(array_seq, pad_value = 0):
  if MAX_SENT_LENGTH>len(array_seq):
      fixed_length_array = np.append(array_seq, np.zeros(MAX_SENT_LENGTH-len(array_seq)))
  else:
      fixed_length_array = array_seq[0:MAX_SENT_LENGTH-1]
  return fixed_length_array

def prepare_sequence(seq, to_ix):
  idxs = [to_ix[w] for w in seq]
  return torch.tensor(idxs, dtype=torch.long)

def tag_sentence(test_file, model_file, out_file):
  ##### load test data #####
  #print('Loading data...')
  with open(test_file, 'r') as fp:
      test = fp.readlines()
  print('Loaded test with', len(test), 'samples...')
  
  ##### load model #####
  model = torch.load(model_file)
  model.eval()

  ##### format data #####
  _test = []
  test_batch_size = len(test)
  for row in test:
    # split the sentence into cleaned word-tag units
    row_test = [i.rstrip().lower() for i in row.split(' ')]

    row_test = [
      model.word_to_idx[w] 
      if w in model.word_to_idx.keys() 
      else model.word_to_idx[UNKNOWN_TAG] 
      for w in row_test
      ]
    row_test = pad_sequence(row_test)
    _test.append(row_test)
  sentence_in = torch.tensor(_test, dtype=torch.long).view(test_batch_size, MAX_SENT_LENGTH)

  ##### predict #####
  model.batch_size = test_batch_size
  with torch.no_grad():
    if device != torch.device("cpu"):
      # Move data to GPU if available
      sentence_in = sentence_in.to(device)

    tag_scores, predicted = model(sentence_in)
    predicted = predicted.detach().cpu().numpy()
  
  idx_to_tag = {v: k for k, v in model.tag_to_idx.items()}
  final_output = []
  for pred_tags, row in zip(predicted, test):
    row_test = [i.rstrip() for i in row.split(' ')]
    row_output = [idx_to_tag[i] for i in pred_tags[0:len(row_test)]]
    output = ' '.join([word.rstrip()+'/'+tag for word, tag in zip(row_test, row_output)])
    final_output.append(output)

  ##### save output #####
  with open(out_file, "w") as output:
      output.write('\n'.join(final_output))

  print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
