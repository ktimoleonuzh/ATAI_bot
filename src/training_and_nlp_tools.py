from torch.utils.data import Dataset
import numpy as np
import random
from load_data import (
    header,
    film_entities, 
    all_movies_dict, 
    all_people_dict, 
    special_movies, 
    special_chars,
    indirectSubclassOf_entities,
    ner
    )

from difflib import SequenceMatcher

import spacy
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("entityLinker", last=True)


def token_lem(sentence):
    word_list = [token.lemma_ for token in nlp(sentence) if not token.is_punct]
    return word_list

class EntityRecognition():
    def __init__(self, sentence, graph):
        if sentence[-1] == '?':
            self.sentence = sentence.split('?')[0]
        else:
            self.sentence = sentence
        self.graph = graph
        self.movies, self.people, self.misc = self.find_entities()
        self.linked_entities = self.map_all_entities()
        if self.linked_entities is not None:
            self.word_list = self.token_lem()
        else:
            self.word_list = [token.lemma_ for token in nlp(self.sentence) if (not token.is_punct)&(token.pos_!='PROPN')]
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
            best_match_label = best_match(self.sentence, special_movies)
            best_match_id = get_key_from_value(best_match_label, all_movies_dict)
            special.append(best_match_id)
            print("Special movie detected: {}, {}, {}.".format(best_match_label, best_match_id, self.get_entity_description(best_match_id)))
        # only for indirectSubclassOf predicate
        # look in subject dictionary
        for misc_entity in indirectSubclassOf_entities.values():
            if misc_entity in self.sentence:
                misc_entity_id = get_key_from_value(misc_entity, indirectSubclassOf_entities)
                misc.append(misc_entity_id)
                print("Miscellaneous entity detected: {}, {}, {}.".format(misc_entity, self.get_entity_description(misc_entity_id), indirectSubclassOf_entities[misc_entity_id]))
        # Normal Movie Titles/People
        # 1. spacy NER
        # 2. check if film/person
        print("Checking spacy NER.")
        entities_obj = nlp(self.sentence)._.linkedEntities
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
        entities_ner = ner(self.sentence, aggregation_strategy="simple")
        entities_2 = []
        for entity in entities_ner:
            entities_2.append(entity["word"])
        for entity in entities_2:
            if entity in all_movies_dict.values():
                entity_key = get_key_from_value(entity, all_movies_dict)
                movies_2.append(entity_key)
                print("Movie detected: {}, {}, {}.".format(entity, entity_key, self.get_entity_description(entity_key)))
            elif entity in all_people_dict.values():
                entity_key = get_key_from_value(entity, all_people_dict)
                people_2.append(entity_key)
                print("Person detected: {}, {}, {}.".format(entity, entity_key, self.get_entity_description(entity_key)))
            # in case a The should've been included in the title
            elif "The "+entity in all_movies_dict.values():
                entity = "The "+entity
                entity_key = get_key_from_value(entity, all_movies_dict)
                movies_2.append(entity_key)
                print("Movie detected: {}, {}, {}.".format(entity, entity_key, self.get_entity_description(entity_key)))
            # in case an unnecessary The has been included in the title
            elif "The " in entity:
                if entity.split("The ")[1] in all_movies_dict.values():
                    entity = entity.split("The ")[1]
                    entity_key = get_key_from_value(entity, all_movies_dict)
                    movies_2.append(entity_key)
                    print("Movie detected: {}, {}, {}.".format(entity, entity_key, self.get_entity_description(entity_key))) 
        # Find best option between 2 NER models
        #special_labels = [all_movies_dict[i] for i in special]
        movies_1_labels = [all_movies_dict[i] for i in movies_1]
        movies_2_labels = [all_movies_dict[i] for i in movies_2]
        movies = list()
        if (len(movies_1) == len(movies_2)) & (len(movies_1) != 0):
            movies_best_match = get_key_from_value(best_match(self.sentence, movies_1_labels+movies_2_labels), all_movies_dict)
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
            movies_labels = [all_movies_dict[i] for i in movies]
            print("Movies detected: {}".format(movies_labels))
        else:
            print("No movies detected")
        
        people = list()
        people_1_labels = [all_people_dict[i] for i in people_1]
        people_2_labels = [all_people_dict[i] for i in people_2]

        if (len(people_1) == len(people_2)) & (len(people_1) != 0):
            people_best_match = get_key_from_value(best_match(self.sentence, people_1_labels+people_2_labels), all_people_dict)
            people.append(people_best_match)
        elif len(people_1) > len(people_2):
            people.extend(people_1)
        elif len(people_1) < len(people_2):
            people.extend(people_2)
        else: 
            people = None
        if people is not None:
            people_labels = [all_people_dict[i] for i in people]
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
                linked_entities[movie] = all_movies_dict[movie]
        if self.people is not None:
            for people in self.people:
                linked_entities[people] = all_people_dict[people]
        if self.misc is not None:
            for misc in self.misc:
                linked_entities[misc] = indirectSubclassOf_entities[misc]
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
            word_list.extend([token.lemma_ for token in nlp(self.sentence) if (not token.is_punct)&(token.text not in entities)&(token.pos_!='PROPN')])
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

def bag_of_words(vocabulary, sentence):
    bag = np.zeros(len(vocabulary), dtype=np.float32)
    for word in vocabulary:
        if word in sentence:
            idx = vocabulary.index(word)
            bag[idx] += 1
    return bag

def process_intents(intents):
    vocabulary = []
    documents = []
    classes = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = token_lem(pattern)
            vocabulary.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    vocabulary = sorted(set(vocabulary))
    random.shuffle(documents)
    return vocabulary, documents, classes

def prepare_training_data(vocabulary, documents, classes):
    Xtrain = list()
    ytrain = list()
    for document in documents:
        bag = bag_of_words(vocabulary, document[0])
        Xtrain.append(bag)
        tag = document[1]
        ytrain.append(classes.index(tag))
        
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    
    return Xtrain, ytrain

class ChatDataset(Dataset):
    def __init__(self, Xtrain, ytrain):
        self.n_samples = len(Xtrain)
        self.x_data = Xtrain
        self.y_data = ytrain
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    
    def __len__(self):
        return self.n_samples