import yaml
import os

from src.preprocessing.prepare_data import prepare_data
from src.utils import load_graph, load_json
from src.agent import MyBot

with open('config/nadia_bot_credentials.yaml', 'r') as file:
    credentials = yaml.safe_load(file)

username = credentials['username']
password = credentials['password']
url = credentials['url']

# First check if the data directory exists
# If yes, load the data
# If not, create the data directory and load the data
if not os.path.exists('data'):
    print(f"--- Data not found. Building the data... ---")
    os.makedirs('data')
    prepare_data()
else:
    print(f"--- Data found. Loading the data... ---")
    # TODO: verify data loading

graph = load_graph('./data/updated_graph.nt')
image_data = load_json('./data/multimedia.json')

# Initialize the bot
mybot = MyBot(username, password, url, graph, image_data)
# mybot.listen()

