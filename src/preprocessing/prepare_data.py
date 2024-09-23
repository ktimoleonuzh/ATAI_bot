import pickle
import os
from rdflib.namespace import RDFS
from src.question_handling.crowd_questions import Crowd_Response
from src.utils import (
    load_data_config,
    download_file,
    unzip_file,
    load_graph,
    load_pickle,
    save_pickle
)
from src.global_variables import (
    film_entities,
    special_chars,
    WD, WDT, SCHEMA, RDFS, DDIS,
    header
)

def download_graph():
    data_config = load_data_config()
    graph_url = data_config['urls']['ddis_movie_graph_nt']
    graph_dest = data_config['paths']['graph_zip']
    graph_extract_dest = data_config['paths']['data_dir']  # Make sure graph ends up in `data/ddis/`
    download_file(graph_url, graph_dest)
    unzip_file(graph_dest, graph_extract_dest)

def download_embeddings():
    data_config = load_data_config()
    embeddings_url = data_config['urls']['ddis_graph_embeddings']
    embeddings_dest = data_config['paths']['embeddings_zip']
    embeddings_extract_dest = os.path.join(data_config['paths']['data_dir'], 'embeddings')  # Unzip into `embeddings/`
    os.makedirs(embeddings_extract_dest, exist_ok=True)
    download_file(embeddings_url, embeddings_dest)
    unzip_file(embeddings_dest, embeddings_extract_dest)

def download_image_data():
    data_config = load_data_config()
    image_data_url = data_config['urls']['images']
    image_data_dest = data_config['paths']['images_zip']
    download_file(image_data_url, image_data_dest)
    # Save the extracted json directly to data_dir
    unzip_file(image_data_dest, data_config['paths']['data_dir'])

def download_crowd_data():
    data_config = load_data_config()
    crowd_data_url = data_config['urls']['crowd_data_tsv']
    crowd_data_dest = data_config['paths']['crowd_data']
    download_file(crowd_data_url, crowd_data_dest)
    print(f"Downloaded crowd data to {crowd_data_dest}.")

# Generate movie-related predicates
def find_movie_predicates(film_entities, graph=None):
    if graph is None:
        graph = load_graph(data_config['paths']['graph'])
    data_config = load_data_config()
    movie_preds = set()
    predicate_dict = {}

    for film_entity in film_entities.values():
        # Find predicates related to movies
        movies = set(graph.subjects(WDT['P31'], WD[film_entity]))
        for movie in movies:
            movie_preds.update(set(graph.predicates(movie, None)))
        
        # Create dictionary for the labels of all movie-related predicates
        for uri in movie_preds:
            pred_label = None
            if uri in WDT:
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
            elif uri in SCHEMA:
                pred = 'schema:' + str(uri).split('schema.org/')[-1]
                pred_label = str(uri).split('schema.org/')[-1]
            elif uri in DDIS:
                pred = 'ddis:' + str(uri).split('atai/')[-1]
                pred_label = str(uri).split('atai/')[-1]
            elif uri in RDFS:
                pred = 'rdfs:' + str(uri).split('rdf-schema#')[-1]
                pred_label = str(uri).split('rdf-schema#')[-1]
            
            if pred_label:
                predicate_dict[pred] = pred_label
    
    # save the predicate dictionary to a pickle
    save_pickle(predicate_dict, data_config['paths_processed']['predicate_dict'])

# Update predicate dictionary with crowd-sourced data
def update_predicate_dict_with_crowd_data(graph = None):
    if graph is None:
        graph = load_graph(data_config['paths']['graph'])
    data_config = load_data_config()
    predicate_dict = load_pickle(data_config['paths_processed']['predicate_dict'])
    with open(data_config['paths_processed']['aggr_ans_dict'], 'rb') as handle:
        aggr_ans_dict = pickle.load(handle)
    if graph is None:
        graph = load_graph(data_config['paths']['graph'])
    crowd_predicates = {}
    
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
                query_pred_label = header + f"""
                SELECT ?lbl
                WHERE {{
                    wdt:{pred_label} rdfs:label ?lbl.
                }}
                """
                result = list(graph.query(query_pred_label))
                value = result[0][0].toPython() if result else pred_label
            elif pred in DDIS: 
                key = 'ddis:' + pred_label
                value = pred_label
            elif pred in SCHEMA:
                key = 'schema:' + pred_label
                value = pred_label
            elif pred in RDFS:
                key = 'rdfs:' + pred_label
                value = pred_label
        
        if key not in predicate_dict:
            predicate_dict[key] = value
            crowd_predicates[key] = value
    
        # Save updated predicate and crowd-sourced data to pickles
        save_pickle(predicate_dict, data_config['paths_processed']['predicate_dict'])
        save_pickle(crowd_predicates, data_config['paths_processed']['crowd_predicates'])

def generate_label_mappings(graph=None):
    """Generate entity-to-label and label-to-entity mappings from the graph."""
    if graph is None:
        graph = load_graph(data_config['paths']['graph'])
    data_config = load_data_config()
    try:
        ent2lbl = {ent: str(lbl) for ent, lbl in graph.subject_objects(RDFS.label)}
        lbl2ent = {lbl: ent for ent, lbl in ent2lbl.items()}
        # Save the generated mappings to pickles
        with open(data_config['paths_processed']['ent2lbl'], 'wb') as handle:
            pickle.dump(ent2lbl, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(data_config['paths_processed']['lbl2ent'], 'wb') as handle:
            pickle.dump(lbl2ent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Error generating label mappings: {e}")
        return None, None

def generate_special_movies(movies):
    data_config = load_data_config()
    special_movies = [movie for movie in movies if any(char in movie for char in special_chars)]
    save_pickle(special_movies, data_config['paths_processed']['special_movies'])

def prepare_data():
    # Download required files
    download_graph()
    download_embeddings()
    download_image_data()
    download_crowd_data()

    # Load configurations
    data_config = load_data_config()
    # Load the RDF graph
    graph = load_graph(data_config['paths']['graph'])
    # Process movie-related predicates
    find_movie_predicates(film_entities, graph)
    # Update predicate dictionary with crowd-sourced data
    # update_predicate_dict_with_crowd_data(graph) # TODO: not working yet
    # Generate entity-to-label and label-to-entity mappings
    generate_label_mappings(graph)
    # Generate special movies
    generate_special_movies(film_entities)
