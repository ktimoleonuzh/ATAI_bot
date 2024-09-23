import time
import atexit
import requests
import random
import json
import torch
from collections import defaultdict
from src.nlp_utils import EntityRecognition, setup_answer_classifier_model
from src.question_handling.factual_questions import Query_Response
from src.question_handling.multimedia_questions import Multimedia_Response
from src.question_handling.recommendation_questions import Rec_Response
from src.utils import load_graph, load_json, load_resources, load_data_config
from src.training.train import bag_of_words

listen_freq = 2

class MyBot:
    def __init__(self, username, password, url):
        self.username = username
        self.password = password
        self.url = url
        self.agent_details = self.login()
        self.session_token = self.agent_details['sessionToken']
        self.chat_state = defaultdict(lambda: {'messages': defaultdict(dict), 'initiated': False, 'my_alias': None})
        atexit.register(self.logout)

    def setup(self):
        self.data_config = load_data_config()
        print(f"- Loading all necessary data for bot {self.username}...")
        self.kg_graph = load_graph(self.data_config['paths_processed']['updated_graph'])
        self.image_data = load_json(self.data_config['paths']['images'])
        print(f"- Setting up the answer classifier model for bot {self.username}...")
        self.model, self.device, self.vocabulary, self.tags = setup_answer_classifier_model()
        print(f"-  Setting up all relevant NLP resources for bot {self.username}...")
        self.nlp, self.ner = load_resources()

    def listen(self):
        print(f"- Bot {self.username} is now listening for new messages...")
        while True:
            # check for all chatrooms
            current_rooms = self.check_rooms()['rooms']
            for room in current_rooms:
                # ignore finished conversations
                if room['remainingTime'] > 0:
                    room_id = room['uid']
                    if not self.chat_state[room_id]['initiated']:  # check whether the chatroom has been initiated (default: False)
                        # send a welcome message and get the alias of the agent in the chatroom
                        self.post_message(room_id=room_id, message='Welcome! I\'m here to answer all your movie-related questions.')
                        self.chat_state[room_id]['initiated'] = True
                        self.chat_state[room_id]['my_alias'] = room['alias']

                    # check for all messages
                    all_messages = self.check_room_state(room_id=room_id, since=0)['messages']

                    for message in all_messages:
                        if message['authorAlias'] != self.chat_state[room_id]['my_alias']:  # make sure that you're not echoing your own message

                            # check if the message is new
                            if message['ordinal'] not in self.chat_state[room_id]['messages']:  # check if the message has been previously "logged" to the chatroom
                                self.chat_state[room_id]['messages'][message['ordinal']] = message  # "log" the message to the chatrooms history
                                print('\t- Chatroom {} - new message #{}: \'{}\' - {}'.format(room_id, message['ordinal'], message['message'], self.get_time()))
                                response = self.get_response(message['message'])
                                self.post_message(room_id=room_id, message=response)
                        
            time.sleep(listen_freq)

    def login(self):
        agent_details = requests.post(url=self.url + "/api/login", json={"username": self.username, "password": self.password}).json()
        print('- User {} successfully logged in with session \'{}\'!'.format(agent_details['userDetails']['username'], agent_details['sessionToken']))
        return agent_details

    def check_rooms(self):
        return requests.get(url=self.url + "/api/rooms", params={"session": self.session_token}).json()

    def check_room_state(self, room_id: str, since: int):
        return requests.get(url=self.url + "/api/room/{}/{}".format(room_id, since), params={"roomId": room_id, "since": since, "session": self.session_token}).json()

    def post_message(self, room_id: str, message: str):
        tmp_des = requests.post(url=self.url + "/api/room/{}".format(room_id),
                                params={"roomId": room_id, "session": self.session_token}, data=message.encode('utf-8')).json()
        if tmp_des['description'] != 'Message received':
            print('\t\t Error: failed to post message: {}'.format(message))

    def get_time(self):
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())

    def logout(self):
        if requests.get(url=self.url + "/api/logout", params={"session": self.session_token}).json()['description'] == 'Logged out':
            print('- Session \'{}\' successfully logged out!'.format(self.session_token))

    def get_response(self, message):
        with open('./data/intents.json', 'r') as json_data:
            intents = json.load(json_data)

        ER = EntityRecognition(
            message,
            self.kg_graph,
            self.nlp,
            self.ner
        )
        linked_entities = ER.linked_entities
        word_list = ER.word_list
        X = bag_of_words(self.vocabulary, word_list)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)
        
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

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
                            response_obj = Rec_Response(self.kg_graph, linked_entities, intent_responses)
                            response = response_obj.build_response()
                        elif tag == "multimedia":
                            response_obj = Multimedia_Response(self.kg_graph, linked_entities, self.image_data)
                            response = response_obj.build_response()
                        else:
                            intent_responses = intent['responses']
                            response_obj = Query_Response(tag, self.kg_graph, linked_entities, intent_responses, message)
                            response = response_obj.build_response()
                    except:
                        response = "Sorry, could you rephrase your message?"
        else:
            response = "Sorry, could you rephrase your message?"
        print(response)
        return response
