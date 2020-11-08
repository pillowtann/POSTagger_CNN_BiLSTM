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
torch.set_printoptions(profile="full")
start_time = datetime.datetime.now()

##### CONSTANTS/SETTINGS #####
# reproducibility
start_time = datetime.datetime.now()
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic = True

# data set-up
IGNORE_CASE = True
VAL_PROP = 0.2
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
MAX_WORD_LENGTH = 30
NUM_EPOCHS = 10

CNN_IN_CHNL = 1
CNN_OUT_CHNL = 10
CNN_KERNEL = MAX_WORD_LENGTH
CNN_PAD = 1
CNN_STRIDE = 1
POOL_KERNEL = 3
POOL_STRIDE = 1

EMBEDDING_DIM = 1000
HIDDEN_DIM = 500
BATCH_SIZE = 10

DROPOUT_RATE = 1e-2 # increase slightly to try
MOMENTUM = 0.85 # increase to try (up to 0.9)
WEIGHT_DECAY = 1e-4 # not used (unless adam)
LEARNING_RATE = 5e-2

# preferences
USE_CPU = False # default False, for overwriting
BUILDING_MODE = False
REVIEW_MODE = False
if BUILDING_MODE:
  TOTAL_DATA = 20
else:
  TOTAL_DATA = 30000
PRINT_ROUND = max(int(round(round((TOTAL_DATA/BATCH_SIZE)/20, 0)/5,0)*5),1)
# print('Will print for iters in multiples of', PRINT_ROUND)
MINS_TIME_OUT = 9

# gpu
if torch.cuda.is_available() and not USE_CPU:
    device = torch.device("cuda:0")
    # print("Running on the GPU")
else:
    device = torch.device("cpu")
    # print("Running on the CPU")

def prepare_sequence(seq):
    return torch.tensor(seq, dtype=torch.long)

def pad_sequence(seq_list, pad_value = PAD_IDX, max_seq_length = MAX_SENT_LENGTH):
  if max_seq_length>len(seq_list):
      fixed_length_list = seq_list + [PAD_IDX]*(max_seq_length-len(seq_list))
  else:
      fixed_length_list = seq_list[0:max_seq_length]
  return fixed_length_list

def pad_tensor(tensor, pad_value = PAD_IDX, max_seq_length = MAX_SENT_LENGTH):
  if max_seq_length>tensor.shape[0]:
    pads = torch.tensor(np.zeros(max_seq_length-tensor.shape[0]), dtype=torch.long)
    fixed_length_tensor = torch.cat((tensor, pads), dim=0)
  else:
    fixed_length_tensor = tensor[0:max_seq_length]
  return fixed_length_tensor

class FormatDataset(Dataset):
  def __init__(self, characters_in, processed_in, processed_out, word_to_ix, tag_to_ix):
    self.characters_in = characters_in
    self.processed_in = processed_in
    self.processed_out = processed_out
    self.word_to_ix = word_to_ix
    self.tag_to_ix = tag_to_ix
          
  def __len__(self):
    # Run multiple rows at once, i.e. reduce enumerate
    return len(self.processed_in)

  def __getitem__(self, index):
    # Step 2. Get our inputs ready for the network, that is, turn them into
    # Tensors of word indices.
    chars_in = torch.tensor(self.characters_in[index], dtype=torch.float)
    sentence_in = prepare_sequence(self.processed_in[index])
    target_out = prepare_sequence(self.processed_out[index])
    
    return chars_in, sentence_in, target_out

class CNNLSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, pad_idx, dropout_rate, 
    cnn_in_chnl, cnn_out_chnl, cnn_padding, cnn_stride, cnn_kernel_size, pool_kernel_size, pool_stride):
        super(CNNLSTMTagger, self).__init__()
        # values
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size # not used
        self.tagset_size = tagset_size # not used
        self.dropout_rate = dropout_rate # not used
        self.pad_idx = pad_idx
        self.cnn_in_chnl = cnn_in_chnl # not used
        self.cnn_out_chnl = cnn_out_chnl # not used
        self.cnn_padding = cnn_padding # not used
        self.cnn_stride = cnn_stride # not used
        self.cnn_kernel_size = cnn_kernel_size # not used
        self.pool_kernel_size = pool_kernel_size # not used
        self.pool_stride = pool_stride # not used

        # layers
        self.conv1 = nn.Conv1d(cnn_in_chnl, cnn_out_chnl, stride=cnn_stride, kernel_size=cnn_kernel_size, padding=cnn_padding)
        self.max_pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.relu = nn.ReLU()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim-cnn_out_chnl)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.dropout = nn.Dropout(dropout_rate)
        

    def forward(self, chars_in, sentence, orig_seq_lengths):

        batch_size, seq_len, word_len = chars_in.size()
        # print('chars_in.shape:', chars_in.shape) # torch.Size([5, 150, 30])
        # print(chars_in[0])
        cnn_in = chars_in.contiguous().view(batch_size*seq_len, -1, word_len)
        # print('cnn_in.shape:', cnn_in.shape) # torch.Size([750, 1, 30])
        # print(cnn_in[0:5])
        cnn_out = self.conv1(cnn_in)
        # print('cnn_out.shape:', cnn_out.shape)
        # print(cnn_out[0:5])
        pool_out = self.max_pool(self.relu(cnn_out))
        # print('pool_out.shape:', pool_out.shape)
        # print(pool_out[0:20])
        cnn_embeds = pool_out.reshape(batch_size, seq_len, self.cnn_out_chnl)
        # print('cnn_embeds.shape:', cnn_embeds.shape)
        # print(cnn_embeds[0])
        embeds = self.word_embeddings(sentence)
        # print('embeds.shape:', embeds.shape)
        embeds = embeds.contiguous()
        # print('embeds.shape:', embeds.shape)
        embeds = torch.cat([embeds, cnn_embeds], dim=2)
        # print('embeds.shape:', embeds.shape)
        input_x = embeds.transpose(1,0)
        # print('input_x.shape:', input_x.shape)
        packed_input = pack_padded_sequence(input_x, orig_seq_lengths, batch_first=False)
        # print('packed_input.data.shape:', packed_input.data.shape)
        packed_output, (ht, ct) = self.lstm(packed_input)
        # print('packed_output.data.shape:', packed_output.data.shape)
        lstm_out, input_sizes = pad_packed_sequence(packed_output, total_length=seq_len, batch_first=False)
        # print('lstm_out.shape:', lstm_out.shape)
        lstm_dropout = self.dropout(lstm_out.contiguous())
        # print('lstm_dropout.shape:', lstm_dropout.shape)
        tag_space = self.hidden2tag(lstm_dropout)
        # print('tag_space.shape:', tag_space.shape)
        tag_scores = F.log_softmax(tag_space, dim=2)
        # print('tag_scores.shape:', tag_scores.shape)
        tag_scores = tag_scores.permute(1,2,0).contiguous()
        # print('tag_scores.shape:', tag_scores.shape)

        return tag_scores

def load_data_to_generators(train_file):
    """Load data from specified path into train and validate data loaders

    Args:
        train_file (str): [description]

    Returns:
        training_generator:
        validation_generator:
        word_to_ix:
        tag_to_ix: 
    """
    ##### load train data #####
    # print('Loading data...')
    with open(train_file, 'r') as fp:
        data = fp.readlines()
    # print('Loaded train with', len(data), 'samples...')

    ### TRIAL RUN REDUCE DATASET ###
    if BUILDING_MODE:
        print('Building mode activated...')
        data = data[0:TOTAL_DATA].copy()

    # initialise dict
    char_to_ix = {PADDING_TAG: PAD_IDX, UNKNOWN_TAG: 1}
    word_to_ix = {PADDING_TAG: PAD_IDX, UNKNOWN_TAG: 1}
    tag_to_ix = {PADDING_TAG: PAD_IDX}

    # initialise processed lists
    tokenized_chars = [] # cnn model input
    tokenized_words = [] # lstm model input
    tokenized_tags = [] # model output
    length_of_sequences = []

    for row in data:
        data_row = row.split(' ')
        length_of_sequences.append(len(data_row))

        _tokn_row_chars = []
        _tokn_row_words = []
        _tokn_row_tags = []
            
        for wordtag in data_row:
            # split by '/' delimiter from the back once
            # e.g. Invest/Net/NNP causing problems with regular split
            _word, tag = wordtag.rsplit('/', 1)

            ##### rules to format words #####
            # remove trailing newline \n
            _word = _word.rstrip()
            tag = tag.rstrip()
            
            # optional lowercase
            if IGNORE_CASE:
                word = _word.lower() 
            
            # if word is numeric // [do blind] + tag is CD
            if (re.sub(r'[^\w]', '', word).isdigit()):
                word = '<NUM>'
                chars = ['<NUM>']
            else:
                chars = list(_word)

            ##### store #####
            # add to char dict if new
            for ch in [c for c in set(chars) if c not in char_to_ix.keys()]:
                char_to_ix[ch] = len(char_to_ix)
            
            # add to word dict if new
            if word not in word_to_ix.keys():
                word_to_ix[word] = len(word_to_ix)

            # add to tag dict if new
            if tag not in tag_to_ix.keys():
                tag_to_ix[tag] = len(tag_to_ix)

            # add tokenized to sequence
            _tokn_row_chars.append(pad_sequence([char_to_ix[c] for c in chars], 
            pad_value = PAD_IDX, max_seq_length = MAX_WORD_LENGTH)) # list to list
            _tokn_row_words.append(word_to_ix[word]) # item to list
            _tokn_row_tags.append(tag_to_ix[tag]) # item to list
        
        # pad_sequence bug, workaround
        if len(_tokn_row_chars)<MAX_SENT_LENGTH:
            _tp_chwords = _tokn_row_chars + [[PAD_IDX]*MAX_WORD_LENGTH]*(MAX_SENT_LENGTH-len(_tokn_row_chars))
        else:
            _tp_chwords = _tokn_row_chars[0:MAX_SENT_LENGTH-1]
        tokenized_chars.append(_tp_chwords) # list to list # list of list to list
        # tokenized_chars.append(
        #     pad_sequence(_tokn_row_chars,
        #     pad_value = [PAD_IDX]*MAX_WORD_LENGTH, max_seq_length = MAX_SENT_LENGTH)) # list of list to list
        tokenized_words.append(
            pad_sequence(_tokn_row_words,
            pad_value = PAD_IDX, max_seq_length = MAX_SENT_LENGTH)) # list to list
        tokenized_tags.append(
            pad_sequence(_tokn_row_tags,
            pad_value = PAD_IDX, max_seq_length = MAX_SENT_LENGTH)) # list to list

    tokenized_chars = np.array(tokenized_chars)
    tokenized_words = np.array(tokenized_words)
    tokenized_tags = np.array(tokenized_tags)

    # print('tokenized_chars.shape', tokenized_chars.shape)
    # print('tokenized_words.shape', tokenized_words.shape)
    # print('tokenized_tags.shape', tokenized_tags.shape)

    if BUILDING_MODE:
        # Randomly sample
        idx = [i for i in range(len(tokenized_words))]
        random.seed(1234)
        random.shuffle(idx)
        split_idx = round(len(data)*VAL_PROP)
        _train_idx = idx[split_idx:]
        _val_idx = idx[0:split_idx]

        # Generators
        training_set = FormatDataset(tokenized_chars[_train_idx], tokenized_words[_train_idx], tokenized_tags[_train_idx], word_to_ix, tag_to_ix)
        training_generator = DataLoader(training_set, batch_size=BATCH_SIZE, **GENERATOR_PARAMS)

        val_batch_size = len(_val_idx)
        validation_set = FormatDataset(tokenized_chars[_train_idx], tokenized_words[_val_idx], tokenized_tags[_val_idx], word_to_ix, tag_to_ix)
        validation_generator = DataLoader(validation_set, batch_size=val_batch_size, **GENERATOR_PARAMS)

        return training_generator, validation_generator, char_to_ix, word_to_ix, tag_to_ix
  
    else:
        # Generators
        training_set = FormatDataset(tokenized_chars, tokenized_words, tokenized_tags, word_to_ix, tag_to_ix)
        training_generator = DataLoader(training_set, batch_size=BATCH_SIZE, **GENERATOR_PARAMS)

        return training_generator, None, char_to_ix, word_to_ix, tag_to_ix

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

    TIME_OUT = False

    training_generator, validation_generator, char_to_ix, word_to_ix, tag_to_ix = load_data_to_generators(train_file)
    model = CNNLSTMTagger(
        EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), PAD_IDX, DROPOUT_RATE,
        CNN_IN_CHNL, CNN_OUT_CHNL, CNN_PAD, CNN_STRIDE, CNN_KERNEL, POOL_KERNEL, POOL_STRIDE
    ).to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index = PAD_IDX).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    results_dict = {}
    for epoch in range(NUM_EPOCHS):
        print('Epoch {}'.format(epoch), '-'*80)
        results = {'train_loss': [], 'train_acc': []}
        
        for idx, (chars_in, sentence_in, target_out) in enumerate(training_generator):

            if datetime.datetime.now() - start_time > datetime.timedelta(minutes=MINS_TIME_OUT, seconds=40):
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
            chars_in = torch.narrow(chars_in[perm_idx], dim=1, start=0, length=max_seq_len).to(device)
            sentence_in = torch.narrow(sentence_in[perm_idx], dim=1, start=0, length=max_seq_len).to(device)
            target_out = torch.narrow(target_out[perm_idx], dim=1, start=0, length=max_seq_len).to(device)
            
            model.zero_grad()

            tag_scores = model(chars_in, sentence_in, seq_lengths).to(device)
            predicted = torch.argmax(tag_scores, dim=1).detach().cpu().numpy()

            accuracy = calc_accuracy(predicted, target_out, BATCH_SIZE)

            loss = loss_function(tag_scores, target_out)
            loss.backward()
            optimizer.step()

            results['train_loss'].append(loss.data.item())
            results['train_acc'].append(accuracy)

            if idx%PRINT_ROUND==0:
                print('Step {} | Training Loss: {}, Accuracy: {}%'.format(idx, round(loss.data.item(),3), round(accuracy*100,3)))
        
        if TIME_OUT:
            break

        results_dict[epoch] = results

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'word_to_ix': deepcopy(word_to_ix),
        'tag_to_ix': deepcopy(tag_to_ix),
        'char_to_ix': deepcopy(char_to_ix),
        'embedding_dim': deepcopy(model.embedding_dim),
        'hidden_dim': deepcopy(model.hidden_dim),
        'dropout_rate': deepcopy(model.dropout_rate),
        'pad_idx': deepcopy(model.pad_idx),
        'cnn_in_chnl': deepcopy(model.cnn_in_chnl),
        'cnn_out_chnl': deepcopy(model.cnn_out_chnl),
        'cnn_padding': deepcopy(model.cnn_padding),
        'cnn_stride': deepcopy(model.cnn_stride),
        'cnn_kernel_size': deepcopy(model.cnn_kernel_size),
        'pool_kernel_size': deepcopy(model.pool_kernel_size),
        'pool_stride': deepcopy(model.pool_stride),
        'ignore_case': IGNORE_CASE,
        }, model_file)

    print('Time Taken: {}'.format(datetime.datetime.now() - start_time), '-'*60)
    print('Finished...')
		
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)