import json 
import numpy as np 
import random
import pickle
import scipy.stats as stats

from tensorflow import keras
import tensorflow as tf
import keras
from keras import models
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

import firebase_admin
from firebase_admin import credentials, firestore
cred = credentials.Certificate("secret key")
app = firebase_admin.initialize_app(cred)
db = firestore.client()
from google.api_core.client_options import ClientOptions
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = models.load_model("assistant_model.h5")
with open('dialogue.json') as file:
    data = json.load(file)
with open('tokenizer.pickle', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
with open('encoder.pickle', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)
      
@app.route("/", methods=["GET"])
def hello():    
    return "Hello! Server is Running"

@app.route("/chatbot", methods=["GET"])
def chatbot_response():    
    user_input = str(request.args['question'])
    user_input = user_input.lower()
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, truncating='post', maxlen=20)
    predictions = model.predict(padded_sequence)
    label_index = np.argmax(predictions)
    pred_label = encoder.inverse_transform([label_index])[0]
    for dialogue in data['dialogue']:
        if dialogue['label'] == pred_label:
            response = np.random.choice(dialogue['responses'])
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)
