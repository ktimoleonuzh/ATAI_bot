# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 15:12:24 2022

@author: Nadia Timoleon
"""
import numpy as np
import random
from sklearn.metrics import pairwise_distances
from load_data import (
    ent2lbl, 
    id2ent, 
    ent2id,
    entity_emb,
    WD
    )

class Rec_Response():
    def __init__(self, graph, linked_entities, intent_responses):
        self.graph = graph
        self.intent_responses = intent_responses
        self.linked_entities = linked_entities 
        self.movies = self.filter_entities()
        
    def filter_entities(self):
        movies = dict()
        if self.linked_entities == None:
            movies = None
        else:
            check_entities = self.linked_entities.copy()
            for (movie_id, label) in check_entities.items():
                descr = self.get_entity_description(movie_id)
                print(f"Movie detected: {label}, {movie_id}, {descr}.")
                movie_emb_id = ent2id[WD[movie_id]]
                movies[label] = [movie_id, movie_emb_id]
        return movies
    
    def get_entity_description(self, entity):
        query = '''
            PREFIX schema: <http://schema.org/>
            PREFIX wd: <http://www.wikidata.org/entity/>
    
            SELECT ?o WHERE{{
                wd:{} schema:description ?o .
                }}'''.format(entity)
        ent_descr = [row[0].toPython() for row in self.graph.query(query)] # the answer is a list of labels
        return ent_descr
    
    def embedding_query(self, movie_emb_id, num_of_answers=1):
        # compute distance to *any* entity
        distances = pairwise_distances(entity_emb[movie_emb_id].reshape(1, -1), entity_emb,).reshape(-1)
        # find most plausible tails
        most_likely = np.argsort(distances)
        embedding_answer = set()
        for idx in most_likely[:num_of_answers]:
            ent = id2ent[idx]
            lbl = ent2lbl[ent]
            if lbl not in self.movies.keys():
                embedding_answer.add(lbl)
        return embedding_answer
    
    def recs_per_movie(self):
        recs = dict()
        for (label, [_, movie_emb_id]) in self.movies.items():
            recs[label] = self.embedding_query(movie_emb_id, 4)
        return recs
    
    def get_answer(self):
        self.recs_dict = self.recs_per_movie()
        recs = set()
        for _, movie_set in self.recs_dict.items():
            recs = recs|movie_set
        final_recs = list(recs)
        return final_recs
    
    def touch_up_intent_response(self, final_answer):
        input_movies = list(self.movies.keys())
        input_movies_string = ('{} and {}'.format(', '.join(input_movies[:-1]), input_movies[-1]))
        answer_string = ('{} and {}'.format(', '.join(final_answer[:-1]), final_answer[-1]))
        response = random.choice(self.intent_responses)
        replace_list = [(a,b) for (a,b) in (['MOVIES', input_movies_string],['ANSWER', answer_string])]
        for (a, b) in replace_list:
            response = response.replace(a,b)
        return response
    
    def build_response(self):
        final_answer = self.get_answer()
        if final_answer is not None:
            response = self.touch_up_intent_response(final_answer)
            return response
        else:
            return "I was unable to find any good recommendations for you. Wanna try another question?"