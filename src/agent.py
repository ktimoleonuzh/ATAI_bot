import time
import atexit
import requests
from collections import defaultdict
from src.bot_response import get_response

listen_freq = 2

class MyBot:
    def __init__(self, username, password, url ,graph, image_data):
        self.agent_details = self.login(username, password, url)
        self.url = url
        self.session_token = self.agent_details['sessionToken']
        self.chat_state = defaultdict(lambda: {'messages': defaultdict(dict), 'initiated': False, 'my_alias': None})
        self.knownledge_graph = graph
        self.image_data = image_data

        atexit.register(self.logout)

    def listen(self):
        while True:
            # check for all chatrooms
            current_rooms = self.check_rooms(session_token=self.session_token)['rooms']
            for room in current_rooms:
                # ignore finished conversations
                if room['remainingTime'] > 0:
                    room_id = room['uid']
                    if not self.chat_state[room_id]['initiated']:  # check whether the chatroom has been initiated (default: False)
                        # send a welcome message and get the alias of the agent in the chatroom
                        self.post_message(room_id=room_id, session_token=self.session_token, message='Welcome! I\'m here to answer all your movie-related questions.')
                        self.chat_state[room_id]['initiated'] = True
                        self.chat_state[room_id]['my_alias'] = room['alias']

                    # check for all messages
                    all_messages = self.check_room_state(room_id=room_id, since=0, session_token=self.session_token)['messages']

                    for message in all_messages:
                        if message['authorAlias'] != self.chat_state[room_id]['my_alias']:  # make sure that you're not echoing your own message

                            # check if the message is new
                            if message['ordinal'] not in self.chat_state[room_id]['messages']:  # check if the message has been previously "logged" to the chatroom
                                self.chat_state[room_id]['messages'][message['ordinal']] = message  # "log" the message to the chatrooms history
                                print('\t- Chatroom {} - new message #{}: \'{}\' - {}'.format(room_id, message['ordinal'], message['message'], self.get_time()))
                                response = get_response(message['message'], self.knownledge_graph, self.image_data)
                                self.post_message(room_id=room_id, session_token=self.session_token, message=response)
                        
            time.sleep(listen_freq)

    def login(self, username: str, password: str, url: str):
        agent_details = requests.post(url=url + "/api/login", json={"username": username, "password": password}).json()
        print('- User {} successfully logged in with session \'{}\'!'.format(agent_details['userDetails']['username'], agent_details['sessionToken']))
        return agent_details

    def check_rooms(self, session_token: str):
        return requests.get(url=self.url + "/api/rooms", params={"session": session_token}).json()

    def check_room_state(self, room_id: str, since: int, session_token: str):
        return requests.get(url=self.url + "/api/room/{}/{}".format(room_id, since), params={"roomId": room_id, "since": since, "session": session_token}).json()

    def post_message(self, room_id: str, session_token: str, message: str):
        tmp_des = requests.post(url=self.url + "/api/room/{}".format(room_id),
                                params={"roomId": room_id, "session": session_token}, data=message.encode('utf-8')).json()
        if tmp_des['description'] != 'Message received':
            print('\t\t Error: failed to post message: {}'.format(message))

    def get_time(self):
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())

    def logout(self):
        if requests.get(url=self.url + "/api/logout", params={"session": self.session_token}).json()['description'] == 'Logged out':
            print('- Session \'{}\' successfully logged out!'.format(self.session_token))
