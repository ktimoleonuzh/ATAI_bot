import rdflib
import time
import yaml
import pickle
import os
import urllib.request
from question_handling.crowd_questions import Crowd_Response
from preprocessing.prepare_pickles import build_pickles

# Utility function to download files
def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    urllib.request.urlretrieve(url, destination)
    print(f"Downloaded {url}.")

# Load configuration file
def load_config():
    with open('config/data_config.yaml', 'r') as file:
        return yaml.safe_load(file)

# Load RDF graph
def load_graph(graph_path):
    empty_graph = rdflib.Graph()
    start_time = time.time()
    graph = empty_graph.parse(graph_path, format='turtle')
    print(f"--- Loaded graph in: {time.time() - start_time} seconds ---")
    return graph

# Download required data
def download_data(config):
    base_url = "https://files.ifi.uzh.ch/ddis/teaching/2023/ATAI/dataset/"
    
    # Download graph
    graph_url = base_url + "ddis-movie-graph.nt.zip"
    graph_dest = config['paths']['graph_zip_path']
    download_file(graph_url, graph_dest)
    
    # Download crowd data
    crowd_data_url = base_url + "crowd_data/crowd_data.tsv"
    crowd_data_dest = config['paths']['crowd_data_path']
    download_file(crowd_data_url, crowd_data_dest)

    # Download aggregate answers
    aggr_ans_url = base_url + "crowd_data/crowd_data_olat_P344FullstopCorrected.tsv"
    aggr_ans_dest = config['paths']['aggr_ans_dict_path']
    download_file(aggr_ans_url, aggr_ans_dest)

    # Extract the graph if needed
    if graph_dest.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(graph_dest, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(graph_dest))
        print(f"Extracted {graph_dest}.")

# Generate movie-related predicates
def find_movie_predicates(graph, film_entities, namespaces):
    movie_preds = set()
    predicate_dict = {}
    header = f"""
    prefix wdt: <{namespaces['WDT']}>
    prefix wd: <{namespaces['WD']}>
    prefix schema: <{namespaces['SCHEMA']}> 
    prefix ddis: <{namespaces['DDIS']}>
    prefix rdfs: <{namespaces['RDFS']}>
    """
    
    for film_entity in film_entities.values():
        # Find predicates related to movies
        movies = set(graph.subjects(rdflib.URIRef(namespaces['WDT'] + 'P31'), rdflib.URIRef(namespaces['WD'] + film_entity)))
        for movie in movies:
            movie_preds.update(set(graph.predicates(movie, None)))
        
        # Create dictionary for the labels of all movie-related predicates
        for uri in movie_preds:
            pred_label = None
            if str(uri).startswith(namespaces['WDT']):
                pred = 'wdt:' + str(uri).split('prop/direct/')[-1]
                query_pred_label = header + f"""
                SELECT ?lbl
                WHERE {{
                    wdt:{pred.split(':')[-1]} rdfs:label ?lbl.
                }}
                """
                result = list(graph.query(query_pred_label))
                if result:
                    pred_label = result[0][0].toPython()
            elif str(uri).startswith(namespaces['SCHEMA']):
                pred = 'schema:' + str(uri).split('schema.org/')[-1]
                pred_label = str(uri).split('schema.org/')[-1]
            elif str(uri).startswith(namespaces['DDIS']):
                pred = 'ddis:' + str(uri).split('atai/')[-1]
                pred_label = str(uri).split('atai/')[-1]
            elif str(uri).startswith(namespaces['RDFS']):
                pred = 'rdfs:' + str(uri).split('rdf-schema#')[-1]
                pred_label = str(uri).split('rdf-schema#')[-1]
            
            if pred_label:
                predicate_dict[pred] = pred_label
    
    return predicate_dict

# Update predicate dictionary with crowd-sourced data
def update_predicate_dict_with_crowd_data(graph, aggr_ans_dict, namespaces, predicate_dict):
    crowd_predicates = {}
    header = f"""
    prefix wdt: <{namespaces['WDT']}>
    prefix wd: <{namespaces['WD']}>
    prefix schema: <{namespaces['SCHEMA']}> 
    prefix ddis: <{namespaces['DDIS']}>
    prefix rdfs: <{namespaces['RDFS']}>
    """
    
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
            if pred.startswith(namespaces['WDT']):
                key = 'wdt:' + pred_label
                query_pred_label = header + f"""
                SELECT ?lbl
                WHERE {{
                    wdt:{pred_label} rdfs:label ?lbl.
                }}
                """
                result = list(graph.query(query_pred_label))
                value = result[0][0].toPython() if result else pred_label
            elif pred.startswith(namespaces['DDIS']): 
                key = 'ddis:' + pred_label
                value = pred_label
            elif pred.startswith(namespaces['SCHEMA']):
                key = 'schema:' + pred_label
                value = pred_label
            elif pred.startswith(namespaces['RDFS']):
                key = 'rdfs:' + pred_label
                value = pred_label
        
        if key not in predicate_dict:
            predicate_dict[key] = value
            crowd_predicates[key] = value
    
    return predicate_dict, crowd_predicates

# Main script logic
def main():
    # Load configurations
    config = load_config()

    # Download required files
    download_data(config)

    # Load the RDF graph
    graph = load_graph(config['paths']['graph_path'])

    # Film entities (Wikidata Q-IDs)
    film_entities = {
        'animated feature film': 'Q29168811',
        'animated film': 'Q202866',
        'film': 'Q11424',
        '3D film': 'Q229390',
        'live-action/animated film': 'Q25110269'
    }

    # Namespaces
    namespaces = config['urls']

    # Process movie-related predicates
    predicate_dict = find_movie_predicates(graph, film_entities, namespaces)

    # Load aggregated answers from crowd
    with open(config['paths']['aggr_ans_dict_path'], 'rb') as handle:
        aggr_ans_dict = pickle.load(handle)

    # Update predicate dictionary with crowd-sourced data
    predicate_dict, crowd_predicates = update_predicate_dict_with_crowd_data(graph, aggr_ans_dict, namespaces, predicate_dict)

    # Save updated predicate and crowd-sourced data
    with open(config['paths']['predicate_dict_path'], 'wb') as handle:
        pickle.dump(predicate_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(config['paths']['crowd_predicates_path'], 'wb') as handle:
        pickle.dump(crowd_predicates, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Generate entity-to-label and label-to-entity mappings
    ent2lbl, lbl2ent = generate_label_mappings(graph)

    # Save the generated mappings to pickles
    with open(config['paths']['ent2lbl_path'], 'wb') as handle:
        pickle.dump(ent2lbl, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(config['paths']['lbl2ent_path'], 'wb') as handle:
        pickle.dump(lbl2ent, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Serialize the updated graph
    graph.serialize(destination=config['paths']['updated_graph_path'], format='turtle')

# Run preparation
def prepare_data():
    build_pickles()  # Prepare the necessary pickles
    main()  # Execute the main logic for the dataset
