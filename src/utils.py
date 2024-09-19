# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:45:08 2022

@author: Nadia Timoleon
"""
import numpy as np
import csv
import pickle
import rdflib
import time
import json

# FOR QUERIES
# Define some prefixes and namespaces
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

# KNOWLEDGE GRAPH LOADING
def load_graph(graph_path, format='turtle'):
    """Load an RDF graph from a file."""
    try:
        start_time = time.time()
        graph = rdflib.Graph().parse(graph_path, format=format)
        print(f"--- Loaded graph in: {time.time() - start_time} seconds ---")
        return graph
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None


# EMBEDDING LOADING
def load_embeddings(entity_emb_path, relation_emb_path, entity_file, relation_file):
    """Load entity and relation embeddings along with their mappings."""
    try:
        entity_emb = np.load(entity_emb_path)
        relation_emb = np.load(relation_emb_path)

        with open(entity_file, 'r') as f:
            ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(f, delimiter='\t')}
            id2ent = {v: k for k, v in ent2id.items()}

        with open(relation_file, 'r') as f:
            rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(f, delimiter='\t')}
            id2rel = {v: k for k, v in rel2id.items()}

        return entity_emb, relation_emb, ent2id, id2ent, rel2id, id2rel

    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None, None, None, None, None, None


# MULTIMEDIA LOADING
def load_json(json_file_path):
    """Load multimedia data from a JSON file."""
    try:
        with open(json_file_path, 'r') as f:
            start_time = time.time()
            data = json.load(f)
            print(f"--- Loaded {json_file_path} in: {time.time() - start_time} seconds ---")
            return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None


# MOVIE ENTITY DEFINITIONS
film_entities = {
    'animated feature film': 'Q29168811',
    'animated film': 'Q202866',
    'film': 'Q11424',
    '3D film': 'Q229390',
    'live-action/animated film': 'Q25110269'
}

special_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ': ', ':', '!', '-']

# LOAD PICKLE FILES
def load_pickle(pickle_file_path):
    """Load data from a pickle file."""
    try:
        with open(pickle_file_path, 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        print(f"Error loading pickle file {pickle_file_path}: {e}")
        return None
