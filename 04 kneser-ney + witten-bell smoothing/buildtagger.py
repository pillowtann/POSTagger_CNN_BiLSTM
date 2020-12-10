# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import csv
import json
import random
from runtagger import predict


def calculate_emission_dict(corpus, D=0.00000075):
    """
    Calculates probaility of seeing current word given current POS tag (P(word|tag))
    Returns emission probabilities dictionary with tags as keys and 
    dictionary of word counts as value {tag:word}
    
    Kneser-Ney smoothing is applied with fixed parameter D
    
    ------
    input
        corpus: list
    output
        dict
    """
    
    # merge list into one
    _train = ' '.join(corpus)
    _train = _train.split(' ')
    _train

    # initialise dict
    emissions = {}

    for row in _train:
        
        # split by '/' delimiter from the back once
        # e.g. Invest/Net/NNP causing problems with regular split
        word, tag = row.rsplit('/', 1)
        
        # remove trailing newline \n
        word = word.rstrip().lower()
        tag = tag.rstrip()
        
        if tag in emissions.keys():
            # add to total count
            emissions[tag]['total'] += 1
            # if tag already exists
            if word in emissions[tag].keys():
                # if word given tag exists, add counter
                emissions[tag][word] += 1  
            else:
                # if word does not exist for given tag, create it
                emissions[tag][word] = 1
        else:
            # if tag does not exist, create it
            emissions[tag] = {word: 1, 'total': 1}
    
#     # weight counts by total count to get probabilities
#     for tag in emissions.keys():
#         # store total value for tag
#         N = emissions[tag]['total']
#         # remove total key per tag
#         emissions[tag].pop('total', None)
#         # count number of seen tags
#         T = len(emissions[tag].keys())
#         if N>0:
#             # for seen word|tags
#             for word in emissions[tag].keys():
#                 emissions[tag][word] /= (N+T)
#             # add unseen tag for unseen words
#             Z = vocab_size - T
#             emissions[tag]['<UNK>'] = T/(Z*(N+T))
    
    # weight counts by total count to get probabilities
    sumT = 0
    for tag in emissions.keys():
        sumT += len(emissions[tag].keys())
    for tag in emissions.keys():
        # store total value for tag
        N = emissions[tag]['total']
        # remove total key per tag
        emissions[tag].pop('total', None)
        # count number of seen tags
        T = len(emissions[tag].keys())
        if N>0:
            # for seen word|tags
            for word in emissions[tag].keys():
                emissions[tag][word] = (emissions[tag][word]-D)/N
            # add unseen tag for unseen words
            A = (D*T)/N
            emissions[tag]['<UNK>'] = A*T/sumT
    
    # return output
    return emissions

def calculate_transition_dict(corpus, tag_types):
    """
    Calculates probaility of seeing current POS tag given previous POS tag (P(tag1|tag0))
    Returns transition probabilities dictionary with previous tags as keys and 
    dictionary of current tag counts as value {tag0:tag1}
    
    Witten-bell smoothing is applied with fixed vocab V
    
    ------
    input
        corpus: list
        tag_type: list
    output
        dict
    """
    # initialise dict
    transitions = {}
    for tag in tag_types:
        transitions[tag]={}
        transitions[tag]['total'] = 0
        for tag2 in tag_types:
            transitions[tag][tag2] = 0

    # for each sentence
    for row in corpus:
        # split the sentence into word-tag units
        _train = row.split(' ')
        # split units into word and tags explicitly
        _train = [(i.rsplit('/', 1)) for i in _train]
        # keep only tags
        _train = [tag.rstrip() for word, tag in _train]

        for s_idx in range(len(_train)):
            transitions[_train[s_idx-1]]['total']+=1
            if s_idx == 0:
                # if start of sentence
                transitions["<s>"]['total']+=1
                transitions["<s>"][_train[s_idx]]+=1
            else:
                # middle and end of sentence
                transitions[_train[s_idx-1]][_train[s_idx]]+=1
            # no elif because if sentence has 1 word we want both start/end tags to work
            if s_idx == len(_train)-1:
                # if end of sentence
                transitions[_train[s_idx]]["</s>"]+=1
                
    # get total possible number of tags
    vocab_size = len(transitions.keys())
        
    # weight counts by total count to get probabilities
    for tag0 in transitions.keys():
        # store total value for tag
        N = transitions[tag0]['total']
        # remove total key per tag
        transitions[tag0].pop('total', None)
        # count number of seen tags
        T = sum([1 for i in transitions[tag].values() if i>0])
        Z = vocab_size - T
        if N>0:
            for tag1 in transitions[tag0].keys():
                # for seen tag1|tag
                if transitions[tag0][tag1]>0:
                    transitions[tag0][tag1] /= (N+T)
                # for UNseen tag1|tag
                else:
                    transitions[tag0][tag1] = T/(Z*(N+T))
                
    return transitions

def evaluate(out_lines, ref_lines):
    """
    Calculates accuracy of two lists of tagged sentences.
    
    ------
    out_lines
        list
    ref_lines
        dict
    """
    total_tags = 0
    matched_tags = 0
    
    for i in range(0, len(out_lines)):
        cur_out_line = out_lines[i].strip()
        cur_out_tags = cur_out_line.split(' ')
        cur_ref_line = ref_lines[i].strip()
        cur_ref_tags = cur_ref_line.split(' ')
        total_tags += len(cur_ref_tags)

        for j in range(0, len(cur_ref_tags)):
            if cur_out_tags[j] == cur_ref_tags[j]:
                matched_tags += 1
    
    acc = float(matched_tags) / total_tags
    print("Accuracy=", acc)
    
    return acc
    
def train_model(train_file, model_file, cross_val = True):
    """
    Main function to train model
    
    ------
    train_file
        str
    model_file
        str
    cross_val
        bool
    """
    
    ##### load train data #####
    print('Loading data...')
    with open(train_file, 'r') as fp:
        train = fp.readlines()
    print('Loaded train with', len(train), 'samples...')
    
    ##### calculate emission probabilities #####
    emissions = calculate_emission_dict(train)
    
    ##### calculate transition probabilities #####
    # get list of possible tags
    tag_types = list(emissions.keys())
    # print status to console
    print(len(tag_types),'tags found in data...')
    # add start and end of sentence markers
    tag_types.extend(["<s>","</s>"])
    # calculations
    transitions = calculate_transition_dict(train, tag_types)
    
    ##### IMRPOVE emission probabilities #####
    accuracy = {}
    D_values = []
    if cross_val:
        # initialise param
        D=0.75
        # number of epochs
        for i in range(0,10):
            D_values.append(D)
            print('Epoch',i,'...')
            random.seed(123)
            random.shuffle(train)
            split_idx = min(round(len(train)*0.2),50)
            _train = train[split_idx:]
            _val = train[0:split_idx]
            _emissions = calculate_emission_dict(_train, D)
            stripped_val = []
            for row in _val:
                # split the sentence into word-tag units
                _r_val = row.split(' ')
                # split units into word and tags explicitly
                _r_val = [(i.rsplit('/', 1)) for i in _r_val]
                # keep only words
                _r_val = [word.rstrip() for word, tag in _r_val]
                # join into sentence
                stripped_val.append(' '.join(_r_val))
            # remove tags from val
            pred = predict(_emissions, transitions, stripped_val)
            accuracy[i] = evaluate(pred, _val)
            # iteratively try to set smaller D values
            if i>2: # patience
                if (accuracy[i]<accuracy[i-1]):
                    D/=5
                elif (accuracy[i]<accuracy[i-1]) & (accuracy[i]<accuracy[i-2]):
                    break
                else:
                    D/=10

            else:
                D/=10
                
    best_D = D_values[max(accuracy, key=accuracy.get)]
    emissions = calculate_emission_dict(train, best_D)
    print('Final parameter D value is:', D)
                    
    ##### save dictionaries #####
    print('Saving dictionaries...')
    with open(model_file, 'w') as fp:
        json.dump({
            'emis': emissions,
            'tran': transitions
        }, fp)
    
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
