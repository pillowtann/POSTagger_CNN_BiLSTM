# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import csv
import json


def calculate_emission_dict(corpus):
    """
    Calculates probaility of seeing current word given current POS tag (P(word|tag))
    Returns emission probabilities dictionary with tags as keys and 
    dictionary of word counts as value {tag:word}
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
    
    # weight counts by total count to get probabilities
    for tag in emissions.keys():
        # store total value for tag
        N = emissions[tag]['total']
        # remove total key per tag
        emissions[tag].pop('total', None)
        if N>0:
            for word in emissions[tag].keys():
                emissions[tag][word] /= N
        
    # return output
    return emissions

def calculate_transition_dict(corpus, tag_types):
    """
    Calculates probaility of seeing current POS tag given previous POS tag (P(tag1|tag0))
    Returns transition probabilities dictionary with previous tags as keys and 
    dictionary of current tag counts as value {tag0:tag1}
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
                
    # weight counts by total count to get probabilities
    for tag0 in transitions.keys():
        # store total value for tag
        N = transitions[tag0]['total']
        # remove total key per tag
        transitions[tag0].pop('total', None)
        if N>0:
            for tag1 in transitions[tag0].keys():
                transitions[tag0][tag1] /= N
                
    return transitions

def train_model(train_file, model_file):
    
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
