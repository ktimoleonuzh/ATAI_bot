# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:57:03 2022

@author: Nadia Timoleon
"""
from rdflib.term import Literal
from src.preprocessing.preprocess_crowddata import namespace_map


class Crowd_Response:
    def __init__(self, task):
        self.task = task
        self.triple = self.get_triple()
        self.answerId = self.task["AnswerId"]
        self.correction = self.get_correction()
        self.distribution = self.task["Distribution"]
        self.fleiss_kappa = self.task["FleissKappa"]
        self.HITId = self.task["HITId"]
    
    def get_triple(self):
        triple = self.task["Triple"]
        subj = triple[0]
        pred = triple[1]
        obj = triple[2]
        return (add_namespace(subj), add_namespace(pred), add_namespace(obj))

    def get_correction(self):
        if self.answerId == 2:
            correction = self.task["Correction"]
            if correction is not None:
                subj = correction[0]
                pred = correction[1]
                obj = correction[2]
                return (add_namespace(subj), add_namespace(pred), add_namespace(obj))
            else:
                return None
        else:
            return None

def add_namespace(string):
    if len(string.split(':')) == 1:
        return Literal(string)
    else:
        ns = namespace_map[string.split(':')[0]]
        item = string.split(':')[1]
        return ns[item]