import yaml
import os

from src.preprocessing.prepare_data import prepare_data
from src.utils import load_credentials, load_training_config
from src.training.train import train_model
from src.agent import MyBot

# First check if the data directory exists
# If not, create the data directory and load the data
# If yes, check that all the necessary files are present
if not os.path.exists('data'):
    print(f"--- Data not found. Building the data... ---")
    os.makedirs('data')
    prepare_data()
else:
    pass
    # TODO: verify data loading

# Then, check if the classifier model exists
# If not, train the model and save it
model_path = load_training_config()['model_path']
if not os.path.exists(model_path):
    print(f"--- Classifier odel not found. Training the model... ---")
    train_model()
else:
    print(f"--- Classifier model found. ---")

# Initialize the bot
username, password, url = load_credentials()
mybot = MyBot(username, password, url)
mybot.listen()

