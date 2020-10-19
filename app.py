import numpy as np
import torch
import torch.nn as nn
from preprocessing import tokenize, bag_of_words
from torch.utils.data import Dataset, DataLoader
from model_name import NeuralNet

import random, json, itertools, io, uuid, datetime
from flask import Flask, jsonify, request

Load intents
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

#Load data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

device = torch.device('cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device) 
model.load_state_dict(model_state)
model.eval()

def Placeholder(sentence, model):
    while True:
        # sentence = "do you use credit cards?
        # sentence and make probabilities
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    if tag == "BMI":
                        return "BMI của bạn là {}".format(bmi)
                    else:
                        return random.choice(intent['responses'])
                    
        else:
            return "Tôi không hiểu bạn nói gì..."

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.data.decode('utf-8')
        dict = json.loads(data)
        createdAt = dict["createdAt"]
        message_user = dict["text"]
        return jsonify({
        "text": Placeholder(message_user, model),
        "createdAt": createdAt,
        "user": { "_id": "2", "name": "Placeholder", "avatar": "https://image.flaticon.com/icons/png/512/2040/2040946.png" },
        "_id": uuid.uuid4()})

if __name__ == '__main__':
    app.run()