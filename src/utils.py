# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:45:08 2022

@author: Nadia Timoleon
"""
import csv
import os
import pickle
import rdflib
import time
import json
import yaml
import spacy
import zipfile
import urllib.request
import numpy as np
from transformers import pipeline
from tqdm import tqdm

# Utility function to download files, with progress bar
def download_file(url, destination):
    # Open the URL connection
    with urllib.request.urlopen(url) as response:
        # Get the total file size
        total_size = int(response.info().get('Content-Length').strip())
        # Open the destination file in write-binary mode
        with open(destination, 'wb') as out_file:
            # Set up tqdm progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination, ascii=True) as progress_bar:
                # Read the data in chunks
                chunk_size = 1024
                while True:
                    data = response.read(chunk_size)
                    if not data:
                        break
                    out_file.write(data)
                    progress_bar.update(len(data))

def unzip_file(zip_file, destination_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination_dir)
    print(f"Downloaded and extracted file to {destination_dir}.")
    # Optionally delete the zip file
    os.remove(zip_file)

def load_resources(nlp_only=False):
    """Load all models and dictionaries.""" 
    # Load the Spacy model and add the entity linker
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)
    # Specify the exact model and revision for the NER pipeline
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    revision = "f2482bf"
    # Create the NER pipeline with the specified model and revision
    ner = pipeline('ner', model=model_name, revision=revision)
    
    if nlp_only:
        return nlp
    else:
        return nlp, ner

# KNOWLEDGE GRAPH LOADING
def load_graph(graph_path, format='turtle'):
    """Load an RDF graph from a file."""
    try:
        start_time = time.time()
        print(f"--- Loading graph from {graph_path} ---")
        graph = rdflib.Graph().parse(graph_path, format=format)
        print(f"--- Loaded graph in: {time.time() - start_time} seconds ---")
        return graph
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None

# EMBEDDING LOADING
def load_embeddings():
    """Load entity and relation embeddings along with their mappings."""
    data_config = load_data_config()
    entity_emb_path = data_config['paths_embeddings']['entity_emb']
    relation_emb_path = data_config['paths_embeddings']['relation_emb']
    entity_file = data_config['paths_embeddings']['entity_file']
    relation_file = data_config['paths_embeddings']['relation_file']
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

# CONFIGURATION LOADING
def load_data_config():
    """Load configuration data from a YAML file."""
    try:
        with open('config/data_config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

# TRAINING CONFIG
def load_training_config():
    """Load training configuration from a YAML file."""
    try:
        with open('config/training_config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading training configuration: {e}")
        return None
    
# CREDENTIALS LOADING
def load_credentials():
    """Load credentials from a YAML file."""
    try:
        with open('config/nadia_bot_credentials.yaml', 'r') as f:
            credentials = yaml.safe_load(f)
        username = credentials['username']
        password = credentials['password']
        url = credentials['url']
        return username, password, url
    except Exception as e:
        print(f"Error loading credentials: {e}")
        return None

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


# SAVE PICKLE FILES
def save_pickle(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# LOAD PICKLE FILES
def load_pickle(pickle_file_path):
    """Load data from a pickle file."""
    try:
        with open(pickle_file_path, 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        print(f"Error loading pickle file {pickle_file_path}: {e}")
        return None
