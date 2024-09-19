# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 21:36:50 2022

@author: Nadia Timoleon
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training.model import NeuralNet
from training_and_nlp_tools import (
    process_intents, 
    prepare_training_data, 
    ChatDataset
)
    
intents = json.loads(open('./data/intents.json').read())
vocabulary, documents, classes = process_intents(intents)
Xtrain, ytrain = prepare_training_data(vocabulary, documents, classes)    

batch_size = 10
input_size = len(vocabulary)
hidden_size = 10
output_size = len(classes)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset(Xtrain, ytrain)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"vocabulary": vocabulary,
"tags": classes
}

model_file = "./training_data/model.pth"
torch.save(data, model_file)

