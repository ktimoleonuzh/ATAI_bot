import rdflib
import time
import yaml
import pickle
from question_handling.crowd_questions import Crowd_Response
from preprocessing.prepare_pickles import build_pickles

def load_config():
    with open('config/data_config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_graph(graph_path):
    empty_graph = rdflib.Graph()
    start_time = time.time()
    graph = empty_graph.parse(graph_path, format='turtle')
    print(f"--- Loaded graph in: {time.time() - start_time} seconds ---")
    return graph

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

def update_kg_with_crowd_data(graph, aggr_ans_dict):
    fixed_triples = {}
    added_triples = {}
    
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
    
    return fixed_triples, added_triples

def process_indirectSubclassOf(graph, aggr_ans_dict, namespaces):
    indirectSubclassOf_triples = {}
    indirectSubclassOf_entities = {}

    header = f"""
    prefix wdt: <{namespaces['WDT']}>
    prefix wd: <{namespaces['WD']}>
    prefix schema: <{namespaces['SCHEMA']}> 
    prefix ddis: <{namespaces['DDIS']}>
    prefix rdfs: <{namespaces['RDFS']}>
    """
    
    for task in aggr_ans_dict['crowddata']:
        crowd_ans = Crowd_Response(task)
        if crowd_ans.answerId == 1:
            triple = crowd_ans.triple
            subj = triple[0].split('/')[-1]
            pred = triple[1]
            obj = triple[2].split('/')[-1]
            if pred == namespaces['DDIS'] + 'indirectSubclassOf':
                query_subj_label = header + f"""
                SELECT ?lbl
                WHERE {{
                    wd:{subj} rdfs:label ?lbl.
                }}
                """
                subj_label = list(graph.query(query_subj_label))[0][0].toPython() if list(graph.query(query_subj_label)) else subj

                query_obj_label = header + f"""
                SELECT ?lbl
                WHERE {{
                    wd:{obj} rdfs:label ?lbl.
                }}
                """
                obj_label = list(graph.query(query_obj_label))[0][0].toPython() if list(graph.query(query_obj_label)) else obj
                
                indirectSubclassOf_triples[subj] = [obj, crowd_ans.HITId]
                indirectSubclassOf_entities[subj] = subj_label
    
    return indirectSubclassOf_triples, indirectSubclassOf_entities

def main():
    # Load configurations
    config = load_config()

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

    # 1. Process movie-related predicates
    predicate_dict = find_movie_predicates(graph, film_entities, namespaces)

    # Load aggregated answers from crowd
    with open(config['paths']['aggr_ans_dict_path'], 'rb') as handle:
        aggr_ans_dict = pickle.load(handle)

    # 2. Update predicate dictionary with crowd-sourced data
    predicate_dict, crowd_predicates = update_predicate_dict_with_crowd_data(graph, aggr_ans_dict, namespaces)

    # 3. Update the knowledge graph (KG) with corrections from crowd data
    fixed_triples, added_triples = update_kg_with_crowd_data(graph, aggr_ans_dict)

    # 4. Process triples with `indirectSubclassOf` predicate
    indirectSubclassOf_triples, indirectSubclassOf_entities = process_indirectSubclassOf(graph, aggr_ans_dict, namespaces)

    # Save updated files
    with open(config['paths']['predicate_dict_path'], 'wb') as handle:
        pickle.dump(predicate_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(config['paths']['crowd_predicates_path'], 'wb') as handle:
        pickle.dump(crowd_predicates, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config['paths']['fixed_triples_path'], 'wb') as handle:
        pickle.dump(fixed_triples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config['paths']['added_triples_path'], 'wb') as handle:
        pickle.dump(added_triples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config['paths']['indirectSubclassOf_triples_path'], 'wb') as handle:
        pickle.dump(indirectSubclassOf_triples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config['paths']['indirectSubclassOf_entities_path'], 'wb') as handle:
        pickle.dump(indirectSubclassOf_entities, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Serialize the updated graph to a file
    graph.serialize(destination=config['paths']['updated_graph_path'], format='turtle')

if __name__ == "__main__":
    build_pickles()  # Run the preparation of pickle files
    main()  # Execute the main logic for predicate processing and updates
