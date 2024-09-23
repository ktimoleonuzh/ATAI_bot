import os

from src.preprocessing.prepare_data import (
    prepare_data,
    download_graph,
    download_embeddings,
    download_image_data,
    download_crowd_data,
    find_movie_predicates,
    generate_label_mappings,
    generate_special_movies
)
from src.utils import load_credentials, load_training_config
from src.global_variables import film_entities
from src.training.train import train_model
from src.agent import MyBot

# First check if the data directory exists
# If not, create the data directory and load the data
# If yes, check that all the necessary files are present
print("--- Checking data directory ---")
if not os.path.exists('data'):
    print(f"--- Data not found. Building the data... ---")
    os.makedirs('data')
    prepare_data()
else:
    # Verify data loading
    # Check for the graph file
    if not os.path.exists('data/ddis/14_graph.nt'):
        print(f"--- Graph file not found. Building the data... ---")
        download_graph()
    # Check for the embeddings file
    if not os.path.exists('data/ddis/embeddings/'):
        print(f"--- Embeddings file not found. Building the data... ---")
        download_embeddings()
    # Check for the image data file
    if not os.path.exists('data/ddis/images.json'):
        print(f"--- Image data file not found. Building the data... ---")
        download_image_data
    # Check for the crowd data file
    if not os.path.exists('data/ddis/crowd_data.tsv'):
        print(f"--- Crowd data file not found. Building the data... ---")
        download_crowd_data()
    # Check for the predicates dictionary
    if not os.path.exists('data/processed/predicate_dict.pkl'):
        print(f"--- Movie predicates dictionary not found. Building the data... ---")
        find_movie_predicates(film_entities)
    # Check for the label mappings
    if not os.path.exists('data/processed/ent2lbl.pkl') or not os.path.exists('data/processed/lbl2ent.pkl'):
        print(f"--- Label mappings not found. Building the data... ---")
        generate_label_mappings()
    # Check for the special movies
    if not os.path.exists('data/processed/special_movies.pkl'):
        print(f"--- Special movies not found. Building the data... ---")
        generate_special_movies(film_entities)
print("--- Data directory check complete ---")

# Then, check if the classifier model exists
# If not, train the model and save it
model_path = load_training_config()['model_path']
if not os.path.exists(model_path):
    print(f"--- Classifier model not found. Training the model... ---")
    train_model()
else:
    print(f"--- Classifier model found. ---")

# Initialize the bot
username, password, url = load_credentials()
mybot = MyBot(username, password, url)
mybot.setup()
mybot.listen()
