# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 21:36:50 2022

@author: Nadia Timoleon
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.training.model import NeuralNet
from src.utils import load_json, load_training_config, load_resources
from src.training.training_dataset import process_intents, ChatDataset

def bag_of_words(vocabulary, sentence):
    bag = np.zeros(len(vocabulary), dtype=np.float32)
    for word in vocabulary:
        if word in sentence:
            idx = vocabulary.index(word)
            bag[idx] += 1
    return bag

def prepare_training_data(vocabulary, documents, classes, batch_size):
    Xtrain = list()
    ytrain = list()
    for document in documents:
        bag = bag_of_words(vocabulary, document[0])
        Xtrain.append(bag)
        tag = document[1]
        ytrain.append(classes.index(tag))
        
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    dataset = ChatDataset(Xtrain, ytrain)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return train_loader

def train_model(
    input_size, hidden_size,
    output_size, vocabulary,
    documents, classes,
    learning_rate, num_epochs,
    batch_size
    ): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = prepare_training_data(vocabulary, documents, classes, batch_size)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'final loss: {loss.item():.4f}')
    return model

def main():
    intents = load_json('./data/intents.json')
    nlp = load_resources(nlp_only=True)
    vocabulary, documents, classes = process_intents(intents, nlp)
    
    # Hyper-parameters: load from config
    input_size = len(vocabulary)
    output_size = len(classes)
    training_config = load_training_config()
    hidden_size = training_config['hidden_size']
    learning_rate = training_config['learning_rate']
    num_epochs = training_config['num_epochs']
    batch_size = training_config['batch_size']
    model_file = training_config['model_file']

    trained_model = train_model(
        input_size, hidden_size,
        output_size, vocabulary,
        documents, classes,
        learning_rate, num_epochs,
        batch_size
    )
    data = {
    "model_state": trained_model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "vocabulary": vocabulary,
    "tags": classes
    }

    torch.save(data, model_file)

    print(f'model saved to {model_file}')

