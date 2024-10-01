import numpy as np
import random
import json
from nltk_utils import  bag_of_word, tokenize, stem
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open ('intents.json','r') as f:
    intents= json.load(f)

all_words=[]
tags=[]
xy=[]

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag =intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w=tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

ignore_words=['?','!','.',',',]  # to remove punkt char
# stem and lower each word
all_words=[stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words=sorted(set(all_words))
tags=sorted(set(tags))

# create training data
X_train=[]
Y_train=[]

for (pattern_sentece ,tag ) in xy: 
    # X: bag of words for each pattern_sentence   
    bag=bag_of_word(pattern_sentece,all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label=tags.index(tag)
    Y_train.append(label) 

X_train=np.array(X_train)
Y_train=np.array(Y_train)

#Hyper-parameters 
num_epochs = 1000
batch_size = 16
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data=X_train
        self.y_data=Y_train
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__ (self,index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset=ChatDataset()
train_loader= DataLoader(dataset=dataset, batch_size=batch_size,shuffle=True)


model=NeuralNet(input_size,hidden_size,output_size)
# Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words=words
        labels=labels
     
        # Forward pass
        outputs=model(words)
        loss=criterion(outputs,labels.long())


        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss:{loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags  
}
FILE='data.pth'
torch.save(data,FILE)

print(f'training complete. file saved to {FILE}')
  
