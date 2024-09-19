import yaml
import os

from preprocessing.prepare_data import prepare_data
from utils import load_graph, load_json
from agent import MyBot

with open('config/nadia_bot_credentials.yaml', 'r') as file:
    credentials = yaml.safe_load(file)

username = credentials['username']
password = credentials['password']

# First check if the data directory exists
# If yes, load the data
# If not, create the data directory and load the data
if not os.path.exists('data'):
    print(f"--- Data not found. Building the data... ---")
    os.makedirs('data')
    prepare_data()
else:
    print(f"--- Data found. Loading the data... ---")

graph = load_graph('./data/updated_graph.nt')
image_data = load_json('./data/multimedia.json')

# Initialize the bot
mybot = MyBot(username, password, graph, image_data)
mybot.listen()

