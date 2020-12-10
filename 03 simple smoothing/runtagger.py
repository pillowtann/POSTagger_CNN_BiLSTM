# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import json


def tag_sentence(test_file, model_file, out_file):

    ##### load test data #####
    print('Loading data...')
    with open(test_file, 'r') as fp:
        test = fp.readlines()
    print('Loaded test with', len(test), 'samples...')
    
    ##### load test data #####
    with open(model_file, 'r') as fp:
        dictionary = json.load(fp)
    emissions = dictionary['emis']
    transitions = dictionary['tran']
    tag_types = list(emissions.keys())
    print('Loaded model dicts...')

    ##### predict #####
    final_output = []
    
    # start loop #
    for row in test:
        
        # split the sentence into cleaned word-tag units
        _test = [i.rstrip().lower() for i in row.split(' ')]

        # initialise path probability matrix
        M = {}
        best_seq = []

        for j, word in enumerate(_test):
            M[j] = {}
            for i, tag in enumerate(tag_types):
                # pointer points at previous best path
                # prob is the prob up to the current tag
                M[j][tag] = {'point': None, 'prob': 0}

        # iterate over each word of sequence
        for j, word in enumerate(_test):

            # initialise tag list per word
            step_info = {}

            # if start of sentence (E.g. "he")
            if j==0:
                for tag1 in tag_types:
                    M[j][tag1]['point'] = "<s>"
                    if word in emissions[tag1].keys():
                        M[j][tag1]['prob'] = emissions[tag1][word] * transitions["<s>"][tag1]
                    else:
                        M[j][tag1]['prob'] = 0.0000000001 * transitions["<s>"][tag1]
                        pass

            # if rest of sentence
            else:
                # for each current tag
                for tag1 in tag_types:

                    step_probs = dict.fromkeys(tag_types, 0)

                    # for each alive previous tag
                    for tag0 in tag_types:
                        step_probs[tag0] = transitions[tag0][tag1] * M[j-1][tag0]['prob']

                    # get previous best path given current best
                    best_prev = max(step_probs, key=step_probs.get)

                    # store current step info into total
                    M[j][tag1]['point'] = best_prev

                    if word in emissions[tag1].keys():
                        M[j][tag1]['prob'] = step_probs[best_prev] * emissions[tag1][word]
                    else:
                        M[j][tag1]['prob'] = step_probs[best_prev] * 0.0000000001
                        pass

                    if j==len(_test)-1:
                        M[j][tag1]['prob'] *= transitions[tag1]["</s>"]

        ### end loop ###

        # get last best tag
        final_prob = [M[len(_test)-1][t]['prob'] for t in tag_types]
        pointer = tag_types[final_prob.index(max(final_prob))]

        # tag final sequence
        for j in range(len(_test)-1, -1, -1):
            # add prev tag
            best_seq.append(pointer)
            # update pointer
            pointer = M[j][pointer]['point']

        best_seq.reverse()
        output = ' '.join([word.rstrip()+'/'+tag for word, tag in zip(row.split(' '), best_seq)])
        final_output.append(output) 
        
    # end loop #
    
    ##### save output #####
    with open(out_file, "w") as output:
        output.write('\n'.join(final_output))
    
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
