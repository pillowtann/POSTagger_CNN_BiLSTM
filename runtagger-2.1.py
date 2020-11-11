# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import datetime
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

start_time = datetime.datetime.now()
IGNORE_CASE = True
START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNKNOWN_TAG = "<UNK>"
MAX_WORD_LENGTH = 30
MAX_SENT_LENGTH = 150
TEST_BATCH_SIZE = 10
PAD_IDX = 0

# preferences
USE_CPU = False # default False, can overwrite
BUILDING_MODE = False
REVIEW_MODE = False

# gpu
if torch.cuda.is_available() and not USE_CPU:
  device = torch.device("cuda:0")
#   print("Running on the GPU")
else:
  device = torch.device("cpu")
#   print("Running on the CPU")

class CNNLSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, chars_size, vocab_size, tagset_size, pad_idx, dropout_rate, 
    ch_embedding_dim, cnn_out_chnl, cnn_padding, cnn_stride, cnn_kernel_size, pool_stride, features_dim):
        super(CNNLSTMTagger, self).__init__()
        # values
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.chars_size  = chars_size  # not used
        self.vocab_size = vocab_size # not used
        self.tagset_size = tagset_size # not used
        self.dropout_rate = dropout_rate # not used
        self.pad_idx = pad_idx
        self.ch_embedding_dim = ch_embedding_dim # not used
        self.cnn_out_chnl = cnn_out_chnl # not used
        self.cnn_padding = cnn_padding # not used
        self.cnn_stride = cnn_stride # not used
        self.cnn_kernel_size = cnn_kernel_size # not used
        self.pool_stride = pool_stride
        self.features_dim = features_dim # not used

        # layers

        self.char_embedding = nn.Embedding(chars_size, ch_embedding_dim, padding_idx = pad_idx)
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=ch_embedding_dim,
                      out_channels=cnn_out_chnl[i],
                      kernel_size=cnn_kernel_size[i],
                      padding=cnn_padding,
                      stride=cnn_stride)
            for i in range(len(cnn_kernel_size))
        ])
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim-sum(cnn_out_chnl)-features_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.dropout = nn.Dropout(dropout_rate)
        

    def forward(self, chars_in, feats_in, sentence, orig_seq_lengths):

        batch_size, seq_len, word_len = chars_in.size()
        # print('chars_in.shape:', chars_in.shape) # torch.Size([10, 49, 100])
        # print(chars_in[0])
        chars_in = chars_in.contiguous().view(batch_size*seq_len, word_len)
        # print('chars_in.shape:', chars_in.shape) # torch.Size([490, 1, 100])
        char_embeds = self.char_embedding(chars_in).permute(0, 2, 1)
        # print('char_embeds.shape:', char_embeds.shape) # (batch_size, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(char_embeds)) for conv1d in self.conv1d_list]
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2], stride=self.pool_stride) for x_conv in x_conv_list]
        pool_out = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        # print('pool_out.shape:', pool_out.shape) # (batch_size, sum(num_filters))
        # print(pool_out[0:20])
        cnn_embeds = pool_out.view(batch_size, seq_len, -1)
        # print('cnn_embeds.shape:', cnn_embeds.shape) # torch.Size([10, 49, 196])
        # print(cnn_embeds[0])
        embeds = self.word_embeddings(sentence)
        # print('embeds.shape:', embeds.shape)
        embeds = embeds.contiguous()
        # print('embeds.shape:', embeds.shape)
        # print('feats_in.shape:', feats_in.shape)
        embeds = torch.cat([embeds, cnn_embeds, feats_in.float()], dim=2)
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

def pad_tensor(tensor, pad_value = PAD_IDX, max_seq_length = MAX_SENT_LENGTH):
  if max_seq_length>tensor.shape[0]:
    pads = torch.tensor(np.zeros(max_seq_length-tensor.shape[0]), dtype=torch.long)
    fixed_length_tensor = torch.cat((tensor, pads), dim=0)
  else:
    fixed_length_tensor = tensor[0:max_seq_length]
  return fixed_length_tensor

def pad_sequence(seq_list, pad_value = PAD_IDX, max_seq_length = MAX_SENT_LENGTH):
  if max_seq_length>len(seq_list):
      fixed_length_list = seq_list + [PAD_IDX]*(max_seq_length-len(seq_list))
  else:
      fixed_length_list = seq_list[0:max_seq_length]
  return fixed_length_list

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def tag_sentence(test_file, model_file, out_file):
    ##### load test data #####
    #print('Loading data...')
    with open(test_file, 'r') as fp:
        test = fp.readlines()

    if TEST_BATCH_SIZE is not None:
        test_batch_size = TEST_BATCH_SIZE 
    else:
        # take full data if gpu can support
        test_batch_size = len(test)
    print('Loaded test with', len(test), 'samples...')
  
    ##### load model #####
    # torch.save({
    #     'epoch': epoch + 1,
    #     'state_dict': model.state_dict(),
    #     'optimizer' : optimizer.state_dict(),
    #     'word_to_ix': deepcopy(word_to_ix),
    #     'tag_to_ix': deepcopy(tag_to_ix),
    #     'char_to_ix': deepcopy(char_to_ix),
    #     'embedding_dim': deepcopy(model.embedding_dim),
    #     'hidden_dim': deepcopy(model.hidden_dim),
    #     'dropout_rate': deepcopy(model.dropout_rate),
    #     'pad_idx': deepcopy(model.pad_idx),
    #     'ch_embedding_dim': deepcopy(model.ch_embedding_dim),
    #     'cnn_out_chnl': deepcopy(model.cnn_out_chnl),
    #     'cnn_padding': deepcopy(model.cnn_padding),
    #     'cnn_stride': deepcopy(model.cnn_stride),
    #     'cnn_kernel_size': deepcopy(model.cnn_kernel_size),
    #     'pool_stride': deepcopy(model.pool_stride),
    #     'ignore_case': IGNORE_CASE,
    #     }, model_file)
    # model = CNNLSTMTagger(
    #     EMBEDDING_DIM, HIDDEN_DIM, len(char_to_ix), len(word_to_ix), len(tag_to_ix), PAD_IDX, DROPOUT_RATE,
    #     CH_EMBEDDING_DIM, CNN_OUT_CHNL, CNN_PAD, CNN_STRIDE, CNN_KERNEL, POOL_STRIDE
    # ).to(device)
    checkpoint = torch.load(model_file) # load model
    model = CNNLSTMTagger(
        checkpoint['embedding_dim'], 
        checkpoint['hidden_dim'], 
        len(checkpoint['char_to_ix']),
        len(checkpoint['word_to_ix']),
        len(checkpoint['tag_to_ix']),
        checkpoint['pad_idx'],
        checkpoint['dropout_rate'],
        checkpoint['ch_embedding_dim'],
        checkpoint['cnn_out_chnl'],
        checkpoint['cnn_padding'],
        checkpoint['cnn_stride'],
        checkpoint['cnn_kernel_size'],
        checkpoint['pool_stride'],
        checkpoint['features_dim']
        ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()

    ##### format data #####
    _chars = []
    _feats = []
    _test = []
    for row in test:
        row_chars = []
        row_feats = []
        row_test = []
        for _word in row.split(' '):
            _word = _word.rstrip()
            # optional lowercase
            if checkpoint['ignore_case']:
                word = _word.lower() 
            
            # if word is numeric // [do blind] + tag is CD
            if (re.sub(r'[^\w]', '', word).isdigit()):
                word = '<NUM>'
                chars = ['<NUM>']
            else:
                chars = list(_word)

            # get morphological features (8 features)
            is_ing = 1 if word[-3:] == 'ing' else 0
            is_ed = 1 if word[-2:] == 'ed' else 0
            is_er = 1 if word[-2:] == 'er' else 0
            is_s = 1 if word[-1:] == 's' else 0
            is_ly = 1 if word[-2:] == 'ly' else 0
            un_is = 1 if word[:2] == 'un' else 0
            is_title = 1 if word[0].isupper() else 0
            word_feats = [len(chars), is_ing, is_ed, is_er, is_s, is_ly, un_is, is_title]

            _tp_char = []
            for ch in chars:
                if ch not in checkpoint['char_to_ix'].keys():
                    _tp_char.append(checkpoint['char_to_ix'][UNKNOWN_TAG])
                else:
                    _tp_char.append(checkpoint['char_to_ix'][ch])

            if word not in checkpoint['word_to_ix'].keys():
                row_test.append(checkpoint['word_to_ix'][UNKNOWN_TAG])
            else:
                row_test.append(checkpoint['word_to_ix'][word])
            
            row_chars.append(pad_sequence(_tp_char, pad_value = PAD_IDX, max_seq_length = MAX_WORD_LENGTH))
            row_feats.append(word_feats)
        
        if len(row_chars)<MAX_SENT_LENGTH:
            _tp_chwords = row_chars + [[PAD_IDX]*MAX_WORD_LENGTH]*(MAX_SENT_LENGTH-len(row_chars))
            _tp_features = row_feats + [[PAD_IDX]*checkpoint['features_dim']]*(MAX_SENT_LENGTH-len(row_feats))
        else:
            _tp_chwords = row_chars[0:MAX_SENT_LENGTH-1]
            _tp_features = row_feats[0:MAX_SENT_LENGTH-1]

        _chars.append(_tp_chwords) # list to list # list of list to list
        _feats.append(_tp_features)
        _test.append(pad_sequence(row_test, pad_value = PAD_IDX, max_seq_length = MAX_SENT_LENGTH))
    
    _chars = torch.tensor(_chars, dtype=torch.long).view(len(test), MAX_SENT_LENGTH, MAX_WORD_LENGTH)
    _feats = torch.tensor(_feats, dtype=torch.long).view(len(test), MAX_SENT_LENGTH, checkpoint['features_dim'])
    _test = torch.tensor(_test, dtype=torch.long).view(len(test), MAX_SENT_LENGTH)
    print('_chars.shape:', _chars.shape)
    print('_feats.shape:', _feats.shape)
    print('_test.shape:', _test.shape)
    # print(sentences[0:10])

    predicted = []
    ##### predict #####
    with torch.no_grad():
        # format batch data
        for i in range(0, len(test)//test_batch_size+1):
            chars_in = _chars[i*test_batch_size:(i+1)*test_batch_size]
            feats_in = _feats[i*test_batch_size:(i+1)*test_batch_size]
            sentence_in = _test[i*test_batch_size:(i+1)*test_batch_size]

            if sentence_in.shape[0]<test_batch_size:
                # if not full, add buffer rows
                n_rows_to_add = test_batch_size-sentence_in.shape[0]
                chsent_dummy = _chars[0:n_rows_to_add]
                chars_in = torch.cat((chars_in, chsent_dummy), dim=0)
                print('chars_in.shape:', chars_in.shape)
                feats_dummy = _feats[0:n_rows_to_add]
                feats_in = torch.cat((feats_in, feats_dummy), dim=0)
                print('feats_in.shape:', feats_in.shape)
                sent_dummy = _test[0:n_rows_to_add]
                sentence_in = torch.cat((sentence_in, sent_dummy), dim=0)
                print('sentence_in.shape:', sentence_in.shape)

            # format batch data
            sentence_lengths = []
            for row in sentence_in:
                sentence_lengths.append(torch.nonzero(row).shape[0])
            sentence_lengths = torch.LongTensor(sentence_lengths)
            seq_lengths, perm_idx = sentence_lengths.sort(0, descending=True)
            sentence_in = sentence_in[perm_idx].to(device)
            feats_in = feats_in[perm_idx].to(device)
            chars_in = chars_in[perm_idx].to(device)

            # predict
            tag_scores = model(chars_in, feats_in, sentence_in, seq_lengths).to(device)
            pred = torch.argmax(tag_scores, dim=1).detach().cpu().numpy()
            
            # sort back to original order
            _, unperm_idx = perm_idx.sort(0)
            predicted.extend(pred[unperm_idx])

    predicted = predicted[0:len(test)]
    # print(len(predicted))
    # print(predicted[0:10])
  
    idx_to_tag = {v: k for k, v in checkpoint['tag_to_ix'].items()}
    final_output = []
    for pred_tags, row in zip(predicted, test):
        row_test = [i.rstrip() for i in row.split(' ')]
        row_output = [idx_to_tag[i] for i in pred_tags[0:len(row_test)]]
        output = ' '.join([word.rstrip()+'/'+tag for word, tag in zip(row_test, row_output)])
        final_output.append(output)

    ##### save output #####
    with open(out_file, "w") as output:
        output.write('\n'.join(final_output))

    print('Time Taken: {}'.format(datetime.datetime.now() - start_time), '-'*60)
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
