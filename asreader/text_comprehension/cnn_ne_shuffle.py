__author__ = 'nyx'

import random
import numpy as np

named_entities = None
named_entities_codes = None

"""
Shuffles the anonymized tokens replacing the named entities in each example. That way, the meaning of the anonymous token
cannot be transferred between examples so the model needs to rely only on the context document provided with each question
in order to answer. This is done to match the setup from Deepmind's paper "Teaching Machines to Read and Comprehend"
(http://arxiv.org/abs/1506.03340).
"""

def set_dictionary(dictionary):
    # named entities start with "#"
    global named_entities
    named_entities = filter(lambda word : word.find("@ent") == 0, dictionary)
    global named_entities_codes
    named_entities_codes = map(lambda word : dictionary[word], named_entities)


def shuffle_ne(data):

    context, context_mask, question, question_mask, answer, candidates, candidates_mask = data
    global named_entities_codes
    #shuffle the ne codes
    shuffled_ne = named_entities_codes[:]
    random.shuffle(shuffled_ne)

    #map the ne codes in the batch to the shuffled ones
    # use negative integers for already replaced named entities, this way we avoid accidental multiple overrides
    for i, ne_code in enumerate(named_entities_codes):
        new_ne_code = -shuffled_ne[i]

        context[context == ne_code] = new_ne_code
        candidates[candidates == ne_code] = new_ne_code
        question[question == ne_code] = new_ne_code
        answer[answer == ne_code] = new_ne_code

    # convert all negative indices to positive
    context = np.abs(context)
    candidates = np.abs(candidates)
    question = np.abs(question)
    answer = np.abs(answer)

    # sanity check that looks for duplicate token IDs
    """
    for codes, mask in zip(candidates.tolist(),candidates_mask.tolist()):
        codes_trim = codes[0:int(sum(mask))]
        if len(codes_trim)!=len(set(codes_trim)):
                print "DUPLICATE token present!"
                print str(codes)
    """

    return context, context_mask, question, question_mask, answer, candidates, candidates_mask
