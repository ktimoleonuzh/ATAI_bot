import yaml
import os

from build_data import build_data
from load_data import load_data

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
    build_data()
else:
    print(f"--- Data found. Loading the data... ---")
    load_data()
    
