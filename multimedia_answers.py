# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:58:24 2022

@author: Nadia Timoleon
"""
from load_all_data import namespace_map
import random

class Multimedia_Response():
    
    def __init__(self, graph, linked_entities, image_data):
        self.graph = graph
        self.linked_entities = linked_entities
        self.image_data = image_data
        self.person = self.filter_entities()

    def get_imdb_id(self, entity):
        query = '''
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                PREFIX wd: <http://www.wikidata.org/entity/>
    
                SELECT (str(?imdb) as ?string) WHERE {{
                wd:{} wdt:P345 ?imdb .
                }}
                '''.format(entity)
        imdb_id = [row[0].toPython() for row in self.graph.query(query)]
        return imdb_id
    
    def filter_entities(self):
        person = dict()
        check_entities = self.linked_entities.copy()
        for (entity, label) in check_entities.items():
            descr = self.get_entity_description(entity)
            print(f"Person detected: {label}, {entity}, {descr}.")
            person[entity] = label
        if len(person) > 1:
            print("More than one people detected.")
            return None, None
        elif len(person) == 0:
            person = None
            print("No people detected.")
        return person
        
    def get_entity_description(self, entity):
        query = '''
            PREFIX schema: <http://schema.org/>
            PREFIX wd: <http://www.wikidata.org/entity/>
    
            SELECT ?o WHERE{{
                wd:{} schema:description ?o .
                }}'''.format(entity)
        ent_descr = [row[0].toPython() for row in self.graph.query(query)] # the answer is a list of labels
        return ent_descr 
    
    def person_lookup(self):
        person_imdb = self.get_imdb_id(list(self.person.keys())[0])
        candidate_images = list()
        for elem in self.image_data:
            if (elem['cast'] == person_imdb) & (elem['type'] == 'poster'):
                candidate_images.append(elem['img'])
        # going for any image in case there are no posters
        if not candidate_images:
            for elem in self.image_data:
                if (elem['cast'] == person_imdb):
                    candidate_images.append(elem['img'])
            # if we still cannot find any images, they don't exist
            if not candidate_images:
                image = None
                print("No available images")
            else:
                image = random.choice(candidate_images)
        else:
            image = random.choice(candidate_images)
        return image
    
    def get_image(self):
        if self.person is not None:
            return self.person_lookup()
        else:
            return None            
        
    def build_response(self):
        response = self.get_image()
        if response is not None:
            response = response.replace(".jpg", "")
            return f"image:{response}"
        else:
            return "Could not find any images that correspond to your request. Do you want me to look for something else?"