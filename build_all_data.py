# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:54:05 2022

@author: Nadia Timoleon
"""
import rdflib
import time
from crowd_answers import Crowd_Response
from preprocess_crowddata import preprocess, crowd_data
import pickle

# define some prefixes
WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
DDIS = rdflib.Namespace('http://ddis.ch/atai/')
RDFS = rdflib.namespace.RDFS
SCHEMA = rdflib.Namespace('http://schema.org/')

header = '''
        prefix wdt: <http://www.wikidata.org/prop/direct/>
        prefix wd: <http://www.wikidata.org/entity/>
        prefix schema: <http://schema.org/> 
        prefix ddis: <http://ddis.ch/atai/>'''
        
film_entities = {
    'animated feature flim': 'Q29168811',
    'animated film'        : 'Q202866',
    'film'                 : 'Q11424',
    '3D film'              : 'Q229390',
    'live-action/animated film': 'Q25110269'
}

start_time = time.time()
empty_graph = rdflib.Graph()
graph = empty_graph.parse('./data/14_graph.nt', format='turtle')
print("--- Loaded graph within: %s seconds ---" % (time.time() - start_time))

with open('./data/all_movies_dict.pickle', 'rb') as handle:
    all_movies_dict = pickle.load(handle)
    
with open('./data/all_people_dict.pickle', 'rb') as handle:
    all_people_dict = pickle.load(handle)
    
special_chars = ['0','1','2','3','4','5','6','7','8','9', ': ', ':','!', '-']
special_movies = list()
for movie in all_movies_dict.values():
    if movie is not None:
        for i in movie:
            if i in special_chars:
                special_movies.append(movie)
                break

with open('./data/special_movies.pickle', 'wb') as handle:
    pickle.dump(special_movies, handle, protocol=pickle.HIGHEST_PROTOCOL)



# Find predicates related to movies
movie_preds = set()
predicate_dict = dict()
for film_entity in film_entities.values():
    # Find predicates related to movies
    movies = set(graph.subjects(WDT['P31'], WD[film_entity]))  # P31: is instance of, Q11424: film
    for movie in movies:
        movie_preds.update(set(graph.predicates(movie, None)))
    # Create dictionary for the labels of all movie-related predicates
    header = '''
            prefix wdt: <http://www.wikidata.org/prop/direct/>
            prefix wd: <http://www.wikidata.org/entity/>
            prefix schema: <http://schema.org/> 
            prefix ddis: <http://ddis.ch/atai/>
            prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            '''
    for uri in movie_preds:
        if uri in WDT:
            ns = WDT
            pred = 'wdt:'+uri.split('prop/direct/',1)[-1]
            query_rel_label = header + '''
            SELECT ?lbl
            WHERE{{
                wdt:{} rdfs:label ?lbl.
                }}'''.format(uri.split('prop/direct/',1)[-1])
            label = list(graph.query(query_rel_label))[0][0].toPython()
            predicate_dict[pred] = label
        elif uri in SCHEMA:
            ns = SCHEMA
            label = uri.split('schema.org/',1)[-1]
            pred = 'schema:'+label
            predicate_dict[pred] = label
        elif uri in DDIS:
            ns = DDIS
            label = uri.split('atai/',1)[-1]
            pred = 'ddis:'+label
            predicate_dict[pred] = label
        elif uri in RDFS:
            ns = RDFS
            label = uri.split('rdf-schema#',1)[-1]
            pred = 'rdfs'+label
            predicate_dict[pred] = label


# Load crowd data, remove malicious workers and aggregate based on majority vote
aggr_ans_dict = preprocess(crowd_data)
with open('./data/aggr_ans_dict.pickle', 'wb') as handle:
    pickle.dump(aggr_ans_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# UPDATE PREDICATE DICT
crowd_predicates = dict()
for task in aggr_ans_dict['crowddata']:
    crowd_ans = Crowd_Response(task)
    triple = crowd_ans.triple
    pred = triple[1]
    pred_label = pred.split('/')[-1]
    if pred_label == '.P344':
        pred_label = 'P344'
    if pred_label == 'P520':
        key = 'wdt:' + pred_label
        value = 'armament'
    else:
        if pred in WDT:
            key = 'wdt:' + pred_label
            query_pred_label = header + '''
            SELECT ?lbl
            WHERE{{
                wdt:{} rdfs:label ?lbl.
                }}'''.format(pred_label)
            if len(list(graph.query(query_pred_label))) == 0:
                print(pred)
                value = pred_label
            else:
                value = list(graph.query(query_rel_label))[0][0].toPython()
        elif pred in DDIS:
            key = 'ddis:' + pred_label
            value = pred_label
        elif pred in SCHEMA:
            key = 'schema:' + pred_label
            value = pred_label
        elif pred in RDFS:
            key = 'rdfs:' + pred_label
            value = pred_label
    if key not in predicate_dict.keys():
        predicate_dict[key] = value
        crowd_predicates[key] = value

with open('./data/predicate_dict.pickle', 'wb') as handle:
    pickle.dump(predicate_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('./data/crowd_predicates.pickle', 'wb') as handle:
    pickle.dump(crowd_predicates, handle, protocol=pickle.HIGHEST_PROTOCOL)

# UPDATE KG
#removed_triples = dict()
fixed_triples = dict()
added_triples = dict()
for task in aggr_ans_dict['crowddata']:
    crowd_ans = Crowd_Response(task)
    if crowd_ans.answerId == 2:
        if crowd_ans.triple in graph:
            print('Mistake found in KG. Attempting to fix.')
            if crowd_ans.correction is not None:
                print('Fix found. Removing wrong triple and adding correct one.')
                graph.remove(crowd_ans.triple)
                graph.add(crowd_ans.correction)
                fixed_triples[crowd_ans.HITId] = crowd_ans.correction
    else:
        if crowd_ans.triple not in graph:
            print("New triple detected. Checking for collisions.")
            triple_pattern = (crowd_ans.triple[0], crowd_ans.triple[1], None)
            triple_generator = graph.triples(triple_pattern)
            collision_list = [triple for triple in triple_generator]
            if len(collision_list) == 0:
                print("Adding new triple.")
                graph.add(crowd_ans.triple)
                added_triples[crowd_ans.HITId] = crowd_ans.triple
            else:    
                print("Data-type inconsistency. Fixing for simplicity.")
                for triple in collision_list:
                    graph.remove(triple)
                    graph.add(crowd_ans.triple)

with open('./data/fixed_triples.pickle', 'wb') as handle:
    pickle.dump(fixed_triples, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./data/added_triples.pickle', 'wb') as handle:
    pickle.dump(added_triples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
graph.serialize(destination='./data/updated_graph.nt', format='turtle')


# Handle triples with predicate ddis:indirectSubclassOf in a different way
indirectSubclassOf_triples = dict()
indirectSubclassOf_entities = dict()
for task in aggr_ans_dict['crowddata']:
    crowd_ans = Crowd_Response(task)
    if crowd_ans.answerId == 1:
        triple = crowd_ans.triple
        subj = triple[0].split('/')[-1]
        pred = triple[1]
        obj = triple[2].split('/')[-1]
        if pred == DDIS['indirectSubclassOf']:
            query_subj_label = header + '''
            SELECT ?lbl
            WHERE{{
                wd:{} rdfs:label ?lbl.
                }}'''.format(subj)
            if len(list(graph.query(query_subj_label)))!=0:
                subj_label = list(graph.query(query_subj_label))[0][0].toPython()

            pred_label = pred.split('/')[-1]        

            query_obj_label = header + '''
            SELECT ?lbl
            WHERE{{
                wd:{} rdfs:label ?lbl.
                }}'''.format(obj)
            if len(list(graph.query(query_obj_label)))!=0:
                obj_label = list(graph.query(query_obj_label))[0][0].toPython()
            if crowd_ans.answerId == 1:
                print([subj_label, pred_label, obj_label])

            indirectSubclassOf_triples[subj] = [obj, crowd_ans.HITId]
            indirectSubclassOf_entities[subj] = subj_label
            
with open('./data/indirectSubclassOf_triples.pickle', 'wb') as handle:
    pickle.dump(indirectSubclassOf_triples, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/indirectSubclassOf_entities.pickle', 'wb') as handle:
    pickle.dump(indirectSubclassOf_entities, handle, protocol=pickle.HIGHEST_PROTOCOL)