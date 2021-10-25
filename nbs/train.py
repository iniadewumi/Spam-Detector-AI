
import os
import pathlib, pickle, json
import pandas as pd

from downloader import all_paths
from tensorflow.keras.layers import Dense, Input, Conv1D, MaxPooling1D, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Model, Sequential


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


data = {}

with open(all_paths["metadata"], "rb") as f:
    data = pickle.load(f)

with open(all_paths["tokenizer"], "rb") as f:
    tok = json.load(f)

embed_dim = 128
lstm_out = 196

model = Sequential()
data["X_train"].shape[1]
model.add(Embedding(data["max_words"], embed_dim, input_length=data["X_train"].shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
print(model.summary())


epochs=2
batch_size=32

model.fit(data["X_train"], data["y_train"], validation_data=(data["X_train"], data["y_train"]),  batch_size=batch_size, verbose=True, epochs=epochs)

def predict(text_str, max_words=300, max_sequence=300, tokenizer=None):
    global tok_as_json
    if not tokenizer:
        return "Tokenizer must be provided"
    sequences = tokenizer.texts_to_sequences([text_str])
    x_input = pad_sequences(sequences, maxlen=max_sequence)
    y_output = model.predict(x_input)
    top_y_index = np.argmax(y_output)

    preds = y_output[0]
    tok_as_json = tokenizer.to_json()
    return [{f"{data['map_labels_inv'][str(round(v))]}": k for k, v in enumerate(preds)}]


predict(text_str ="Free Gold! Hello World", max_words=data["max_words"],  max_sequence=data["max_sequence"], tokenizer=data['tokenizer'])


metadata = {
    "max_words": MAX_NUM_OF_WORDS,
    "max_sequence": MAX_SEQ_LEN,
    "map_labels": map_labels,
    "map_labels_inv": map_labels_inv, 
    
}
metadata = json.dumps(metadata)


TOKENIZER_EXPORT_PATH = all_paths["exports"] / "spam-classifier-tokenizer.json"
TOKENIZER_EXPORT_PATH.write_text(tok_as_json)

METADATA_EXPORT_PATH = all_paths["exports"] / "spam-classifier-metadata.json"
METADATA_EXPORT_PATH.write_text(metadata)

MODEL_EXPORT_PATH = all_paths['exports']/'spam-model.h5'
model.save(str(MODEL_EXPORT_PATH))

import boto3
s3 = boto3.client('s3')


files = {"MetadataJSON.json":METADATA_EXPORT_PATH, "Model.h5":MODEL_EXPORT_PATH, "TokenizerJSON.json":TOKENIZER_EXPORT_PATH}
for name, file in files.items():
    with open(str(file), "rb") as f:
        s3.upload_fileobj(f, "ai-tokenizer", "models/"+name)
        

