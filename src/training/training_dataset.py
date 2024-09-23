import random
from torch.utils.data import Dataset

def process_intents(intents, nlp):
    """Process intents for training."""
    vocabulary = []
    documents = []
    classes = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = [token.lemma_ for token in nlp(pattern) if not token.is_punct]
            vocabulary.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    vocabulary = sorted(set(vocabulary))
    random.shuffle(documents)
    return vocabulary, documents, classes

class ChatDataset(Dataset):
    def __init__(self, Xtrain, ytrain):
        self.n_samples = len(Xtrain)
        self.x_data = Xtrain
        self.y_data = ytrain
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    
    def __len__(self):
        return self.n_samples