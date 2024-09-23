import torch
from difflib import SequenceMatcher
from src.training.model import NeuralNet
from src.global_variables import special_chars, film_entities, header
from src.utils import load_pickle, load_training_config, load_data_config

def setup_answer_classifier_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_file = load_training_config()['model_path']
    data = torch.load(model_file, map_location=device, weights_only=True)

    model_state = data["model_state"]
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    vocabulary = data['vocabulary']
    tags = data['tags']

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model, device, vocabulary, tags

class EntityRecognition():
    def __init__(self, sentence, graph, nlp, ner):
        if sentence[-1] == '?':
            self.sentence = sentence.split('?')[0]
        else:
            self.sentence = sentence
        self.graph = graph
        self.nlp = nlp
        self.ner = ner
        self.data_config = load_data_config()
        self.all_movies_dict = load_pickle(self.data_config['paths']['all_movies_dict'])
        self.all_people_dict = load_pickle(self.data_config['paths']['all_people_dict'])
        self.special_movies = load_pickle(self.data_config['paths']['special_movies'])
        self.indirectSubclassOf_entities = load_pickle(self.data_config['paths']['indirectSubclassOf_entities'])
        self.movies, self.people, self.misc = self.find_entities()
        self.linked_entities = self.map_all_entities()
        if self.linked_entities is not None:
            self.word_list = self.token_lem()
        else:
            self.word_list = [token.lemma_ for token in nlp(self.sentence) if (not token.is_punct) & (token.pos_ != 'PROPN')]
            print("No entities detected.")

            
    def find_entities(self):
        # we only append IDs in these lists!
        special = list()
        movies_1, movies_2 = list(), list()
        people_1, people_2 = list(), list()
        misc = list()
        # Special Movie Titles
        # 1. check if special -> MOVIE!
        # 2. find best match
        is_special = False
        for letter in self.sentence:
            if letter in special_chars:
                is_special = True
        if is_special:
            best_match_label = best_match(self.sentence, self.special_movies)
            best_match_id = get_key_from_value(best_match_label, self.all_movies_dict)
            special.append(best_match_id)
            print("Special movie detected: {}, {}, {}.".format(best_match_label, best_match_id, self.get_entity_description(best_match_id)))
        # only for indirectSubclassOf predicate
        # look in subject dictionary
        for misc_entity in self.indirectSubclassOf_entities.values():
            if misc_entity in self.sentence:
                misc_entity_id = get_key_from_value(misc_entity, self.indirectSubclassOf_entities)
                misc.append(misc_entity_id)
                print("Miscellaneous entity detected: {}, {}, {}.".format(misc_entity, self.get_entity_description(misc_entity_id), self.indirectSubclassOf_entities[misc_entity_id]))
        # Normal Movie Titles/People
        # 1. spacy NER
        # 2. check if film/person
        print("Checking spacy NER.")
        entities_obj = self.nlp(self.sentence)._.linkedEntities
        entities_1 = ['Q'+str(entity.get_id()) for entity in entities_obj]
        entities_1_labels = [entity for entity in entities_obj]
        for idx, entity in enumerate(entities_1):
            if self.check_if_film(entity):
                movies_1.append(entity)
                print("Movie detected: {}, {}, {}.".format(entities_1_labels[idx], entity, self.get_entity_description(entity)))
            elif self.check_if_person(entity):
                people_1.append(entity)
                print("Person detected: {}, {}, {}.".format(entities_1_labels[idx], entity, self.get_entity_description(entity)))
            else:
                print("Non-person, non-movie detected: {}, {}, {}.".format(entities_1_labels[idx], entity, self.get_entity_description(entity)))
        # Last Resort
        # If all of the above gives us None
        # 1. transformer NER
        # 2. iterate through people
        # 3. iterate through movies
        print("Checking huggingface NER.")
        entities_ner = self.ner(self.sentence, aggregation_strategy="simple")
        entities_2 = []
        for entity in entities_ner:
            entities_2.append(entity["word"])
        for entity in entities_2:
            if entity in self.all_movies_dict.values():
                entity_key = get_key_from_value(entity, self.all_movies_dict)
                movies_2.append(entity_key)
                print("Movie detected: {}, {}, {}.".format(entity, entity_key, self.get_entity_description(entity_key)))
            elif entity in self.all_people_dict.values():
                entity_key = get_key_from_value(entity, self.all_people_dict)
                people_2.append(entity_key)
                print("Person detected: {}, {}, {}.".format(entity, entity_key, self.get_entity_description(entity_key)))
            # in case a The should've been included in the title
            elif "The "+entity in self.all_movies_dict.values():
                entity = "The "+entity
                entity_key = get_key_from_value(entity, self.all_movies_dict)
                movies_2.append(entity_key)
                print("Movie detected: {}, {}, {}.".format(entity, entity_key, self.get_entity_description(entity_key)))
            # in case an unnecessary The has been included in the title
            elif "The " in entity:
                if entity.split("The ")[1] in self.all_movies_dict.values():
                    entity = entity.split("The ")[1]
                    entity_key = get_key_from_value(entity, self.all_movies_dict)
                    movies_2.append(entity_key)
                    print("Movie detected: {}, {}, {}.".format(entity, entity_key, self.get_entity_description(entity_key))) 
        # Find best option between 2 NER models
        #special_labels = [all_movies_dict[i] for i in special]
        movies_1_labels = [self.all_movies_dict[i] for i in movies_1]
        movies_2_labels = [self.all_movies_dict[i] for i in movies_2]
        movies = list()
        if (len(movies_1) == len(movies_2)) & (len(movies_1) != 0):
            movies_best_match = get_key_from_value(best_match(self.sentence, movies_1_labels+movies_2_labels), self.all_movies_dict)
            movies.append(movies_best_match)
        elif len(movies_1) > len(movies_2):
            movies.extend(movies_1)
        elif len(movies_1) < len(movies_2):
            movies.extend(movies_2)
        if len(special) > 0:
            movies.extend(special)
        if len(movies)==0:
            movies = None
        if movies is not None:
            movies_labels = [self.all_movies_dict[i] for i in movies]
            print("Movies detected: {}".format(movies_labels))
        else:
            print("No movies detected")
        
        people = list()
        people_1_labels = [self.all_people_dict[i] for i in people_1]
        people_2_labels = [self.all_people_dict[i] for i in people_2]

        if (len(people_1) == len(people_2)) & (len(people_1) != 0):
            people_best_match = get_key_from_value(best_match(self.sentence, people_1_labels+people_2_labels), self.all_people_dict)
            people.append(people_best_match)
        elif len(people_1) > len(people_2):
            people.extend(people_1)
        elif len(people_1) < len(people_2):
            people.extend(people_2)
        else: 
            people = None
        if people is not None:
            people_labels = [self.all_people_dict[i] for i in people]
            print("People detected: {}".format(people_labels))
        else:
            print("No people detected.")
        
        if len(misc) == 0:
            misc = None

        return movies, people, misc
    

    def map_all_entities(self):
        linked_entities = dict()
        if self.movies is not None:
            for movie in self.movies:
                linked_entities[movie] = self.all_movies_dict[movie]
        if self.people is not None:
            for people in self.people:
                linked_entities[people] = self.all_people_dict[people]
        if self.misc is not None:
            for misc in self.misc:
                linked_entities[misc] = self.indirectSubclassOf_entities[misc]
        if len(linked_entities) == 0:
            linked_entities = None
        return linked_entities
    
    def check_if_film(self, entity):
        is_film = list()
        for film_type in film_entities.values():
            is_film_query = '''
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                PREFIX wd: <http://www.wikidata.org/entity/>
    
                ASK
                WHERE {{wd:{} wdt:P31 wd:{} .}}
            '''.format(entity, film_type)
            is_film.append(list(self.graph.query(is_film_query))[0])
        return any(is_film)
    
    def check_if_person(self, entity):
        is_person = False
        is_person_query = '''
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
    
            ASK
            WHERE {{wd:{} wdt:P31 wd:Q5 .}}
        '''.format(entity)
        is_person = list(self.graph.query(is_person_query))[0]
        return is_person
    
    def token_lem(self):
        word_list = list()
        for entities in list(self.linked_entities.values()):
            word_list.extend([token.lemma_ for token in self.nlp(self.sentence) if (not token.is_punct)&(token.text not in entities)&(token.pos_!='PROPN')])
        return list(set(word_list))
    
    def get_entity_description(self, entity):
        query = header + '''    
            SELECT ?o WHERE{{
                wd:{} schema:description ?o .
                }}'''.format(entity)
        ent_descr = [row[0].toPython() for row in self.graph.query(query)] # the answer is a list of labels
        return ent_descr
    
def get_key_from_value(value, dictionary):
    key = list(dictionary.keys())[list(dictionary.values()).index(value)]
    return key

def best_match(pattern, candidates):
    best_match_label = None
    best_match_size = 0
    for cand in candidates:
        if cand is not None:
            match = SequenceMatcher(None, pattern, cand).find_longest_match(0, len(pattern), 0, len(cand))
            if match.size > best_match_size:
                best_match_size = match.size
                best_match_label = cand
    return best_match_label

def get_key_from_value(value, dictionary):
    key = list(dictionary.keys())[list(dictionary.values()).index(value)]
    return key
