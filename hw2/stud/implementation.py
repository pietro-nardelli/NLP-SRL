import json
import random

import numpy as np
from typing import List, Tuple

from model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from stud.my_model import MyModel, HParam, Vars
from stud.preprocessing import *
from stud.prediction import *

import numpy as np

def build_model_34(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """
    return StudentModel(device)


def build_model_234(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    # return Baseline(return_predicates=True)
    raise NotImplementedError


def build_model_1234(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    # return Baseline(return_predicates=True)
    raise NotImplementedError


class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, return_predicates=False):
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        for pos in sentence['pos_tags']:
            prob = self.baselines['predicate_identification'][pos]['positive'] / self.baselines['predicate_identification'][pos]['total']
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)

        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(zip(sentence['lemmas'], predicate_identification)):
            if not is_predicate or lemma not in self.baselines['predicate_disambiguation']:
                predicate_disambiguation.append('_')
            else:
                predicate_disambiguation.append(self.baselines['predicate_disambiguation'][lemma])
                predicate_indices.append(idx)

        argument_identification = []
        for dependency_relation in sentence['dependency_relations']:
            prob = self.baselines['argument_identification'][dependency_relation]['positive'] / self.baselines['argument_identification'][dependency_relation]['total']
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(sentence['dependency_relations'], argument_identification):
            if not is_argument:
                argument_classification.append('_')
            else:
                argument_classification.append(self.baselines['argument_classification'][dependency_relation])

        if self.return_predicates:
            return {
                'predicates': predicate_disambiguation,
                'roles': {i: argument_classification for i in predicate_indices},
            }
        else:
            return {'roles': {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path='data/baselines.json'):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, device):
        # Initialize the vocabularies
        self.words_vocabulary = torch.load("model/words_vocabulary.pt")
        self.pos_tags_vocabulary = torch.load("model/pos_tags_vocabulary.pt")
        self.lemmas_vocabulary = torch.load("model/lemmas_vocabulary.pt")
        self.roles_vocabulary = torch.load("model/roles_vocabulary.pt")

        # Initialize the instance of MyModel
        self.model = MyModel(
            device=device,
            input_size=HParam.word_embedding_size +
            HParam.word_embedding_pre_size +
            Vars.predicates_flags_size +
            HParam.pos_tags_embedding_size +
            HParam.lemmas_embedding_size +
            Vars.bert_embedding_size,
            output_size=len(self.roles_vocabulary),
            words_vocabulary=self.words_vocabulary,
            lemmas_vocabulary=self.lemmas_vocabulary,
            pos_tags_vocabulary=self.pos_tags_vocabulary)

        # Send the model to the device
        self.model.to(device)
        self.device = device

        # Load the weights of the model
        self.model.load_state_dict(torch.load("model/model.pt", map_location=torch.device('cpu')), strict=False)


    def predict(self, sentence):
        # Create a vocabulary that maps number to label (this is necessary for generate_predictions)
        roles_reverse_vocabulary = create_roles_reverse_vocabulary(self.roles_vocabulary)
        # Generate the predictions
        return generate_predictions(self.device,
                                    sentence,
                                    self.words_vocabulary,
                                    self.lemmas_vocabulary,
                                    self.pos_tags_vocabulary,
                                    self.model,
                                    roles_reverse_vocabulary)
