# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:56:15 2022

@author: Nadia Timoleon
"""
import numpy as np
import random
from sklearn.metrics import pairwise_distances
from src.training_and_nlp_tools import get_key_from_value, best_match
from src.utils import (
    header,
    WD,
    namespace_map,
    load_embeddings,
    load_pickle
    )

entity_emb, relation_emb, ent2id, id2ent, rel2id, id2rel = load_embeddings(
    './data/ddis-graph-embeddings/entity_embeds.npy',
    './data/ddis-graph-embeddings/relation_embeds.npy',
    './data/ddis-graph-embeddings/entity_ids.del',
    './data/ddis-graph-embeddings/relation_ids.del'
)

crowd_predicates = load_pickle('./data/crowd_predicates.pickle')
all_movies_dict = load_pickle('./data/all_movies_dict.pickle')
predicate_dict = load_pickle('./data/predicate_dict.pickle')
ent2lbl = load_pickle('./data/ent2lbl.pickle')



class Query_Response:
    def __init__(self, tag, graph, linked_entities, intent_responses, sentence):
        self.tag = tag
        self.graph = graph
        self.intent_responses = intent_responses
        if sentence[-1] == '?':
            self.sentence = sentence.split('?')[0]
        else:
            self.sentence = sentence
        self.linked_entities = linked_entities
        self.movie_id, self.movie_label = self.filter_entities()
    
    def filter_entities(self):
        if self.linked_entities == None:
            movie_id, movie_label = None, None
        else:
            check_entities = self.linked_entities.copy()
            if len(list(check_entities.keys())) > 1:
                print("Multiple entities detected.")
                candidates = list(check_entities.values())
                movie_id = get_key_from_value(best_match(self.sentence, candidates), all_movies_dict) 
                movie_label = best_match(self.sentence, candidates) 
                self.linked_endities = dict()
                self.linked_endities[movie_id] = movie_label
            elif len(list(check_entities.keys())) == 1:
                movie_id = list(self.linked_entities.keys())[0]
                movie_label = self.linked_entities[movie_id]
                print("Entity detected: {}, {}, {}".format(movie_id, movie_label, self.get_entity_description(movie_id)))
            else:
                print("No entity detected. Trying exhaustive search.")
        return movie_id, movie_label
    
    def KG_query(self):
        query = header + '''
            SELECT ?query (str(?label) as ?string) WHERE{{
                wd:{} {} ?query.
                ?query rdfs:label ?label .
                }}'''.format(self.movie_id, self.pred)
        if len(list(self.graph.query(query))) != 0:
            KG_answer = [row[1].toPython() for row in self.graph.query(query)] # list of labels   
        else:
            query = header + '''
            SELECT ?query WHERE{{
                wd:{} {} ?query .
                }}'''.format(self.movie_id, self.pred)
            KG_answer = [row[0].toPython() for row in self.graph.query(query)] # list of labels
        return KG_answer
    
    def embedding_query(self, movie_emb, prop_emb, num_of_answers=1):
        # combine according to the TransE scoring function
        lhs = movie_emb + prop_emb
        # compute distance to *any* entity
        distances = pairwise_distances(lhs.reshape(1, -1), entity_emb).reshape(-1)
        # find most plausible tails
        most_likely = np.argsort(distances)
        embedding_answer = list()
        for idx in most_likely[:num_of_answers]:
            ent = id2ent[idx]
            lbl = ent2lbl[ent]
            embedding_answer.append(lbl)
        return embedding_answer
    
    def check_KG_answer(self):
        # Unique answer provided by KG
        if len(self.KG_answer)==1:
            print("KG has the answer.")
            final_answer = self.KG_answer
            self.source = "KG"
        # Multiple answers or no answer in KG
        else:
            if len(self.KG_answer)==0:  # no answer in KG
                print("No answer in KG; looking at embeddings.")
                # simply give the top-1 embedding answer when there is no answer from KG
                movie_emb = entity_emb[ent2id[WD[self.movie_id]]]    
                ns = self.pred.split(":")[0]
                pred = self.pred.split(":")[-1]
                prop_emb = relation_emb[rel2id[namespace_map[ns][pred]]]
                final_answer = self.embedding_query(movie_emb, prop_emb)
                self.source = "Embeddings"
            else:  # more that one answers in KG
                print(f"Multiple answers in KG: {self.KG_answer}.")
                print("Checking embeddings.")                
                movie_emb = entity_emb[ent2id[WD[self.movie_id]]]    
                ns = self.pred.split(":")[0]
                pred = self.pred.split(":")[-1]
                prop_emb = relation_emb[rel2id[namespace_map[ns][pred]]]
                final_answer = self.embedding_query(movie_emb, prop_emb)                
                self.embedding_answer = self.embedding_query(movie_emb, prop_emb, num_of_answers=5)
                if len(self.embedding_answer) == 0:
                    print("Embeddings do not contain that information.")
                    final_answer = self.KG_answer
                    self.source = "KG"
                else:
                    final_answer = self.combine_KG_and_emb()
                    self.source = "KG + Embeddings"
        print(final_answer)
        return final_answer
    
    def combine_KG_and_emb(self):
        # compare to see if we can find an answer that is:
        # 1. different from any given by the KG
        # 2. most recommended by embeddings
        emb_most_rec = self.embedding_answer[0]
        if emb_most_rec not in self.KG_answer:
            print(f"Found top-recommended embedding answer not in KG: {emb_most_rec}")
            final_answer = self.KG_answer + [emb_most_rec]
        else:
            print("Embeddings did not provide any further info.")
            final_answer = self.KG_answer
        return final_answer
    
    def get_answer(self):
        # retrieve predicate based on the tag
        self.pred = list(predicate_dict.keys())[list(predicate_dict.values()).index(self.tag)]
        if self.pred in crowd_predicates.keys():
            print("This question should be delegated to the crowd.")
            #final_answer = None
        if self.movie_id is None or self.pred is None:
            final_answer = None
            print("There was an error, please try rephrasing your question.")
        else:
            self.KG_answer = self.KG_query()
            final_answer = self.check_KG_answer()
        return final_answer
    
    def touch_up_intent_response(self, final_answer):
        if len(final_answer) > 1:
            answer_string = ('{} and {}'.format(', '.join(final_answer[:-1]), final_answer[-1]))
        else: 
            answer_string = final_answer[0]
        response = random.choice(self.intent_responses)
        replace_list = [(a,b) for (a,b) in (['MOVIE', self.movie_label],['ANSWER', answer_string])]
        for (a, b) in replace_list:
            if type(b) != str:
                b = str(b)
            response = response.replace(a,b)
        return response
    
    def build_response(self):
        final_answer = self.get_answer()
        if final_answer is not None:                    
            response = self.touch_up_intent_response(final_answer)
            return response
        else:
            return "I was unable to retrieve the information you asked for. Wanna try another question?"
        
    def get_entity_description(self, entity):
        query = header + '''    
            SELECT ?o WHERE{{
                wd:{} schema:description ?o .
                }}'''.format(entity)
        ent_descr = [row[0].toPython() for row in self.graph.query(query)] # the answer is a list of labels
        return ent_descr
    
def get_URI(item):
    ns = item.split(':')[0]
    ent = item.split(':')[1]
    URI = namespace_map[ns][ent]     
    return URI
