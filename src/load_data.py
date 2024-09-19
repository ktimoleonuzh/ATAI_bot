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
from transformers import pipeline

# Initialize NER pipeline
ner = pipeline('ner')

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

graph = load_graph('./data/updated_graph.nt')

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

entity_emb, relation_emb, ent2id, id2ent, rel2id, id2rel = load_embeddings(
    './data/ddis-graph-embeddings/entity_embeds.npy',
    './data/ddis-graph-embeddings/relation_embeds.npy',
    './data/ddis-graph-embeddings/entity_ids.del',
    './data/ddis-graph-embeddings/relation_ids.del'
)

# Label dictionaries from the knowledge graph
def load_labels(graph):
    """Generate entity-to-label and label-to-entity mappings from the graph."""
    try:
        ent2lbl = {ent: str(lbl) for ent, lbl in graph.subject_objects(RDFS.label)}
        lbl2ent = {lbl: ent for ent, lbl in ent2lbl.items()}
        return ent2lbl, lbl2ent
    except Exception as e:
        print(f"Error generating label mappings: {e}")
        return None, None

ent2lbl, lbl2ent = load_labels(graph)

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

image_data = load_json('./data/images.json')

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

added_triples = load_pickle('./data/added_triples.pickle')
aggr_ans_dict = load_pickle('./data/aggr_ans_dict.pickle')
all_movies_dict = load_pickle('./data/all_movies_dict.pickle')
all_people_dict = load_pickle('./data/all_people_dict.pickle')
crowd_predicates = load_pickle('./data/crowd_predicates.pickle')
fixed_triples = load_pickle('./data/fixed_triples.pickle')
indirectSubclassOf_triples = load_pickle('./data/indirectSubclassOf_triples.pickle')
indirectSubclassOf_entities = load_pickle('./data/indirectSubclassOf_entities.pickle')
predicate_dict = load_pickle('./data/predicate_dict.pickle')
special_movies = load_pickle('./data/special_movies.pickle')

# VERIFY LOADING
def verify_loaded_data():
    """Check if all necessary data is loaded correctly."""
    if None in [graph, entity_emb, relation_emb, ent2id, rel2id, image_data, added_triples, aggr_ans_dict]:
        print("Some data did not load correctly. Please check the logs for errors.")
    else:
        print("All data loaded successfully.")

verify_loaded_data()