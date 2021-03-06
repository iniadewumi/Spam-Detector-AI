from downloader import all_paths
import pandas as pd
import random, pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

full_df_path = all_paths['full_spam_df']
df = pd.read_csv(full_df_path)

labels = df.label.tolist()
texts = df.text.tolist()

map_labels = {'ham':0, 'spam':1}
map_labels_inv = {str(v):k for k, v in map_labels.items()}
labels_int = [map_labels[x] for x in labels]

random_ind = random.randint(0, len(texts))

for _ in range(50):
    assert texts[random_ind] == df.iloc[random_ind].text
    assert map_labels_inv[str(labels_int[random_ind])] == df.iloc[random_ind].label

MAX_NUM_OF_WORDS = 300
tokenizer = Tokenizer(num_words=MAX_NUM_OF_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)


MAX_SEQ_LEN = 300
X = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)


labels_int_array = np.asarray(labels_int)
y = to_categorical(labels_int_array)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=24)

training_data = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "max_words": MAX_NUM_OF_WORDS,
    "max_sequence": MAX_SEQ_LEN,
    "map_labels": map_labels,
    "map_labels_inv": map_labels_inv,
    "tokenizer": tokenizer
}
tokenizer_json = tokenizer.to_json()

with open(all_paths["metadata"], "wb") as f:
    pickle.dump(training_data, f)
all_paths["tokenizer"].write_text(tokenizer_json)