import numpy as np
import torch
from typing import List, Dict

from stud.preprocessing import *


# Return a vocabulary from number to label
def create_roles_reverse_vocabulary(vocab:Dict[str,int]):
    vocabulary = {}
    for i,c in enumerate(vocab):
        vocabulary[i] = c
    vocabulary.update({len(vocabulary.keys()): None}) #None = <PAD>

    return vocabulary


# Predict the roles of a sentence. The output is a list of roles
def predict_roles (device, 
                  words:List[str], 
                  predicates_flags:List[int],
                  lemmas:List[str],  
                  pos_tags:List[str], 
                  bert:List[List[str]], 
                  model, 
                  roles_reverse_vocabulary:Dict[int,str]):
    # No dropout
    model.eval() 
    # No gradient
    with torch.no_grad():
        
        words = torch.LongTensor(words).unsqueeze(0).to(device)
        predicates_flags = torch.FloatTensor(predicates_flags).to(device)
        lemmas = torch.LongTensor(lemmas).unsqueeze(0).to(device)
        pos_tags = torch.LongTensor(pos_tags).unsqueeze(0).to(device)
        # This could be any variable: in this case this is not important because we don't want to use word_dropout
        freq = torch.zeros(0).to(device)
        
        # We need to set bert_prediction = True to avoid to transform a sentence into a window
        pred = model(words, predicates_flags, lemmas, pos_tags, bert, freq, bert_prediction=True).tolist() #predict

        roles_pred = []
        for role in pred:
            role = roles_reverse_vocabulary[np.argmax(role)]
            # The following commented lines of code are used only with the Argument Boundary Technique (they are not used in the final model)
            '''
            if (role == '<BOA>' or role=='<EOA>'):
              role = '_'
            '''
            roles_pred.append(role)
    return roles_pred


# Generate the predictions from a sentence that is a dictionary, like the JSON file
# Returns a dictionary with its main key 'roles' and as sub-keys the index of the predicates
# The arguments for that predicates is a list assigned to the relative index
# If the input sentence has no predicates, the predicate index is not assigned
def generate_predictions (device:str, 
                          sentence:List[str], 
                          words_vocabulary:Dict[str,int], 
                          lemmas_vocabulary:Dict[str,int], 
                          pos_tags_vocabulary:Dict[str,int], 
                          model, 
                          roles_reverse_vocabulary:Dict[int,str]):
    prediction_dict = {}
    roles_pred_dictionary = {}
    # For each predicate in the sentence
    for predicate_index in range(len(sentence['predicates'])):
        # For each predicate, encode the inputs (words, flags, lemmas, ...)
        if (sentence['predicates'][predicate_index] != '_'):
            sentence_length = len(sentence['words'])
            encoded_words = []
            encoded_predicates_flags = []
            encoded_lemmas = []
            encoded_pos_tags = []

            # Encode the words
            for word in sentence['words']:
              encoded_words.append(words_vocabulary.get(word, words_vocabulary['<UNK>']))
            
            # Encoded one hot predicate flags
            one_hot_predicate = np.zeros(len(sentence['words'])).tolist()
            one_hot_predicate[int(predicate_index)] = 1
            encoded_predicates_flags.append(one_hot_predicate)
            
            # Encode lemmas
            for lemma in sentence['lemmas']:
                encoded_lemmas.append(lemmas_vocabulary.get(lemma, lemmas_vocabulary['<UNK>']))

            # Encode pos tags
            for pos_tag in sentence['pos_tags']:
              encoded_pos_tags.append(pos_tags_vocabulary.get(pos_tag, pos_tags_vocabulary['<UNK>']))

            # The bert input is just the sentence as it is included in a list (to produce the right shape)
            bert_sentence = [sentence['words']]
            
            # Predict the roles from the inputs previously defined
            roles_pred = predict_roles(device, encoded_words, encoded_predicates_flags, encoded_lemmas, encoded_pos_tags, bert_sentence, model, roles_reverse_vocabulary)
            # Create a dictionary with predicate index and prediction
            roles_pred_dictionary [predicate_index] = roles_pred

    # If no predicates are found, create an empty dictionary as prediction
    if (not roles_pred_dictionary):

        roles_pred_dictionary = {}

    prediction_dict = {'roles': roles_pred_dictionary}
    return prediction_dict
