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

        # initialise backpointer
        best_seq = []

        # iterate over each word of sequence
        for j, word in enumerate(_test):

            # initialise tag list per word
            tags_probs = dict.fromkeys(tag_types, 0)

            # if start of sentence (E.g. "he")
            if j==0:
                for tag1 in tag_types:
                    if word in emissions[tag1].keys():
                        tags_probs[tag1] = emissions[tag1][word] * transitions["<s>"][tag1]
                    else:
                        tags_probs[tag1] = 0
            # if rest of sentence
            else:
                for tag1 in tag_types:
                    if word in emissions[tag1].keys():
                        tags_probs[tag1] = emissions[tag1][word] * transitions[best_tag][tag1]
                    else:
                        tags_probs[tag1] = 0

            # update best tag before next word
            best_tag = max(tags_probs, key=tags_probs.get)
            # save sequence (optional save of probabilities: tags_probs[best_tag])
            best_seq.append(best_tag)

        # tag final sequence
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
