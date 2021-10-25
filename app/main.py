from fastapi import FastAPI
import pathlib, json
from typing import Optional
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / 'models'
MODEL_PATH = MODELS_DIR / 'Model.h5'
TOKENIZER_PATH = MODELS_DIR / 'tokenizerJSON.json'
METADATA_PATH = MODELS_DIR / 'MetadataJSON.json'


app = FastAPI()


SPAM_MODEL = None
TOKENIZER = None
@app.on_event('startup')
def on_startup():
    print("Loading...")
    if MODEL_PATH.exists():
        global SPAM_MODEL
        SPAM_MODEL = load_model(MODEL_PATH)
    if TOKENIZER_PATH.exists():
        global TOKENIZER
        TOKENIZER = tokenizer_from_json(TOKENIZER_PATH.read_text())

    if METADATA_PATH.exists():
        print("Metadata found\n\n\n\n\n\n")
        global METADATA
        with open(METADATA_PATH, 'r') as f:
            METADATA = json.load(f)
    print("Loaded!")

def predict(query:str):
    sequences = TOKENIZER.texts_to_sequences([query])
    x_input = pad_sequences(sequences, maxlen=METADATA['max_sequence'])
    preds_array = SPAM_MODEL.predict(x_input)
    preds = preds_array[0]


    top_index_value = np.argmax(preds)
    map_labels_inv = METADATA['map_labels_inv']
    
    top_pred = {
        "label": map_labels_inv[str(top_index_value)],
        "confidence": float(preds[top_index_value])
    }

    return  {"top_prediction":top_pred, "predictions": [{"label": map_labels_inv[str(i)], "confidence": float(x)} for i, x in enumerate(preds)]}


@app.get("/")
def read_index(q:Optional[str] = None):
    global METADATA, TOKENIZER, SPAM_MODEL
    query = q or "Hello World!"
    prediction = predict(query)
    print(prediction)
    return {"query": query, "prediction": prediction, "metadata": {**METADATA}}

