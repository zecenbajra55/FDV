import uvicorn
from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
import pickle
import re
import snowballstemmer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained models
model_rnn = load_model('model_rnn.h5')  
model_gru = load_model('model_gru.h5') 
model_lstm = load_model('model_lstm.h5') 

# Define input request body schema using Pydantic BaseModel
# class Item(BaseModel):
#     text: str

# Define preprocessing functions
def clean_text(string):
    if isinstance(string, str):
        clean_text = re.sub(r'[\n,|।\'":]', '', string)
        return clean_text
    else:
        return string  # Return the input unchanged if it's not a string

stop_words = set(stopwords.words('nepali'))  
stemmer = snowballstemmer.NepaliStemmer()

def tokenize_and_stem(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        stemmed = stemmer.stemWords(tokens)
        return ' '.join(stemmed)
    else:
        return ''
    
max_length = 200  # Maximum sequence length
vocab_size=50000
# Load tokenizer and maximum sequence length
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# tokenizer.num_words = 1000  # Set the number of words to keep
# tokenizer.oov_token = '<OOV>'  # Out of vocabulary token

# Labels for prediction
labels = ['business', 'entertainment', 'politics', 'sport', 'tech']

# Define API endpoint for preprocessing and prediction
@app.post("/predict/")
async def predict_item(item_list: list):
    for item in item_list:
    # Clean text
        text_df = pd.Series(item)
        text_df = text_df.apply(clean_text)
        X = text_df.apply(lambda x: tokenize_and_stem(x))

        tokenizer.fit_on_texts(X)
        sequences = tokenizer.texts_to_sequences(X)
        padded_sequences = pad_sequences(sequences, maxlen=max_length)

        prediction_rnn = np.argmax(model_rnn.predict(padded_sequences))
        labels = ['business',  'entertainment', 'politics', 'sport', 'tech']
        predicted_label_rnn = labels[prediction_rnn]

        prediction_gru = np.argmax(model_gru.predict(padded_sequences))
        predicted_label_gru = labels[prediction_gru]

        prediction_lstm = np.argmax(model_lstm.predict(padded_sequences))
        predicted_label_lstm = labels[prediction_lstm]

        
        return {"predicted_label_rnn": predicted_label_rnn,
                "predicted_label_gru": predicted_label_gru,
                "predicted_label_lstm": predicted_label_lstm
                }

# Run the FastAPI app using Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
