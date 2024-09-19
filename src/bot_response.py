# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:34:11 2022

@author: Nadia Timoleon
"""
import random
import json

import torch

from training.model import NeuralNet
from training_and_nlp_tools import bag_of_words, EntityRecognition
from question_handling.factual_questions import Query_Response
from question_handling.multimedia_questions import Multimedia_Response
from question_handling.recommendation_questions import Rec_Response

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./data/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "./training_data/chat_train_data.pth"
data = torch.load(FILE)

model_state = data["model_state"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
vocabulary = data['vocabulary']
tags = data['tags']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def get_response(message, graph, image_data):
    ER = EntityRecognition(message, graph)
    linked_entities = ER.linked_entities
    word_list = ER.word_list
    X = bag_of_words(vocabulary, word_list)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                try:  # making sure that the script won't crash in case something unexpected happens
                    print("Tag detected: {}".format(tag))
                    if tag in ["greeting", "goodbye"]:
                        response = random.choice(intent['responses'])
                    elif tag == "recommendation":
                        intent_responses = intent['responses']
                        response_obj = Rec_Response(graph, linked_entities, intent_responses)
                        response = response_obj.build_response()
                    elif tag == "multimedia":
                        response_obj = Multimedia_Response(graph, linked_entities, image_data)
                        response = response_obj.build_response()
                    else:
                        intent_responses = intent['responses']
                        response_obj = Query_Response(tag, graph, linked_entities, intent_responses, message)
                        response = response_obj.build_response()
                except:
                    response = "Sorry, could you rephrase your message?"
    else:
        response = "Sorry, could you rephrase your message?"
    print(response)
    return response
    