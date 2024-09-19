# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:45:08 2022

@author: Nadia Timoleon
"""
import numpy as np
import csv
import os
import pickle
import rdflib
import time
import json
from transformers import pipeline
ner = pipeline('ner')

# FOR QUERIES
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
        
namespace_map = {
    'wd': WD,
    'wdt': WDT,
    'schema': SCHEMA,
    'ddis': DDIS
}

# KNOWLEDGE GRAPH
start_time = time.time()
empty_graph = rdflib.Graph()
graph = empty_graph.parse('./data/updated_graph.nt', format='turtle')
print("--- Loaded graph within: %s seconds ---" % (time.time() - start_time))

# EMBEDDINGS
# load the embeddings
entity_emb = np.load('./data/ddis-graph-embeddings/entity_embeds.npy')
relation_emb = np.load('./data/ddis-graph-embeddings/relation_embeds.npy')
entity_file = os.path.join('./data/ddis-graph-embeddings/entity_ids.del')
relation_file = os.path.join('./data/ddis-graph-embeddings/relation_ids.del')

# load the dictionaries
with open(entity_file, 'r') as ifile:
    ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
    id2ent = {v: k for k, v in ent2id.items()}
with open(relation_file, 'r') as ifile:
    rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
    id2rel = {v: k for k, v in rel2id.items()}
    
ent2lbl = {ent: str(lbl) for ent, lbl in graph.subject_objects(RDFS.label)}
lbl2ent = {lbl: ent for ent, lbl in ent2lbl.items()}

# MULTIMEDIA
# Opening images meta-data JSON file
f = open('./data/images.json')
# returns JSON object as 
# a dictionary
start_time = time.time()
image_data = json.load(f)
print("--- Loaded image.json within: %s seconds ---" % (time.time() - start_time))


# MY DATA
film_entities = {
    'animated feature flim': 'Q29168811',
    'animated film'        : 'Q202866',
    'film'                 : 'Q11424',
    '3D film'              : 'Q229390',
    'live-action/animated film': 'Q25110269'
}

special_chars = ['0','1','2','3','4','5','6','7','8','9', ': ', ':','!', '-']

# LOAD ALL PICKLE FILES
with open('./data/added_triples.pickle', 'rb') as handle:
     added_triples = pickle.load(handle)
with open('./data/aggr_ans_dict.pickle', 'rb') as handle:
     aggr_ans_dict = pickle.load(handle)
with open('./data/all_movies_dict.pickle', 'rb') as handle:
    all_movies_dict = pickle.load(handle)    
with open('./data/all_people_dict.pickle', 'rb') as handle:
    all_people_dict = pickle.load(handle)     
with open('./data/crowd_predicates.pickle', 'rb') as handle:
     crowd_predicates = pickle.load(handle)
with open('./data/fixed_triples.pickle', 'rb') as handle:
     fixed_triples = pickle.load(handle)
with open('./data/indirectSubclassOf_triples.pickle', 'rb') as handle:
     indirectSubclassOf_triples = pickle.load(handle)
with open('./data/indirectSubclassOf_entities.pickle', 'rb') as handle:
     indirectSubclassOf_entities = pickle.load(handle)  
with open('./data/predicate_dict.pickle', 'rb') as handle:
     predicate_dict = pickle.load(handle)     
with open('./data/special_movies.pickle', 'rb') as handle:
     special_movies = pickle.load(handle)