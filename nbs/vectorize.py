from downloader import all_paths
import pandas as pd
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

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

MAX_NUM_OF_WORDS = 280
tokenizer = Tokenizer(num_words=MAX_NUM_OF_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)


MAX_SEQ_LEN = 300
X = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)


labels_int_array = np.asarray(labels_int)
y = to_categorical(labels_int_array)
