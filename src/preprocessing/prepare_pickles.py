import pickle
import yaml

def load_config():
    with open('config/data_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_pickle(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def prepare_special_movies(movies, special_chars):
    special_movies = [movie for movie in movies if any(char in movie for char in special_chars)]
    return special_movies

def preprocess_crowd_data(aggr_ans_dict, crowd_questions):
    # Implement your preprocess logic here if needed
    return aggr_ans_dict

def build_pickles():
    config = load_config()

    # Special Movies Pickle
    with open(config['paths']['all_movies_dict_path'], 'rb') as handle:
        all_movies_dict = pickle.load(handle)
    
    special_chars = ['0','1','2','3','4','5','6','7','8','9', ': ', ':','!', '-']
    special_movies = prepare_special_movies(all_movies_dict.values(), special_chars)
    save_pickle(special_movies, config['paths']['special_movies_path'])

    # Crowd Data Processing
    with open(config['paths']['aggr_ans_dict_path'], 'rb') as handle:
        aggr_ans_dict = pickle.load(handle)
    
    crowd_predicates = dict()  # Assuming you populate this from crowd processing
    save_pickle(crowd_predicates, config['paths']['crowd_predicates_path'])
    save_pickle(aggr_ans_dict, config['paths']['aggr_ans_dict_path'])

    # Additional Pickles (add your logic for generating these if needed)
    save_pickle({}, config['paths']['fixed_triples_path'])
    save_pickle({}, config['paths']['added_triples_path'])
    save_pickle({}, config['paths']['indirectSubclassOf_triples_path'])
    save_pickle({}, config['paths']['indirectSubclassOf_entities_path'])

if __name__ == '__main__':
    build_pickles()
