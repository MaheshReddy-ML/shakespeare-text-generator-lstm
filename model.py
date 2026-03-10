"""
Shakespeare Text Generator using Bidirectional LSTM

This script trains a language model on Shakespeare dialogue.
The model learns to predict the next word in a sequence and
can generate Shakespeare-style text from a given seed phrase.
"""

import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Input
from tensorflow.keras.layers import Dropout as dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


"""
Load and preprocess the Shakespeare dataset.
"""
print("Loading data...")
df = pd.read_csv("Shakespeare_data.csv")

df = df[['PlayerLine']]
df = df.dropna()
df = df[df['PlayerLine'].str.len() > 20]


"""
Tokenize the text and create a vocabulary.
"""
print("Tokenizing text...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['PlayerLine'])

total_words = len(tokenizer.word_index) + 1

index_word = {index: word for word, index in tokenizer.word_index.items()}


"""
Generate n-gram sequences from each line of text.
"""
print("Creating sequences (this might take a moment)...")
input_sequences = []

for line in df['PlayerLine']:

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i + 1]
        input_sequences.append(n_gram_seq)


"""
Pad sequences so they all have the same length.
"""
max_sequence_len = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_sequence_len,
    padding='pre'
)


"""
Split sequences into input features and labels.
"""
X = input_sequences[:, :-1]
y = input_sequences[:, -1]


"""
Build the Bidirectional LSTM language model.
"""
print("Building the model...")

model = Sequential([

    Input(shape=(max_sequence_len - 1,)),

    Embedding(total_words, 200),

    Bidirectional(LSTM(256, return_sequences=True)),
    dropout(0.2),

    Bidirectional(LSTM(128)),
    dropout(0.2),

    Dense(total_words, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


"""
Define training callbacks for early stopping and learning rate reduction.
"""
early_stop = EarlyStopping(
    monitor='accuracy',
    patience=6,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,
    patience=2,
    min_lr=0.0001,
    verbose=1
)


"""
Train the model on the prepared sequences.
"""
print("\nStarting training...")

model.fit(
    X,
    y,
    epochs=60,
    batch_size=128,
    callbacks=[early_stop, reduce_lr]
)


"""
Save the trained model and tokenizer for later use.
"""
model.save("shakespeare_generator.keras")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved successfully.")


"""
Generate text using the trained language model.
"""

def generate_text(seed_text, next_words=15):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding='pre'
        )

        predicted = model.predict(token_list, verbose=0)

        predicted_word_index = np.argmax(predicted)

        output_word = index_word.get(predicted_word_index, "")

        seed_text += " " + output_word

    return seed_text


"""
Interactive loop for generating Shakespeare-style text.
"""
print("\nShakespeare Generator Ready!")
print("Type a word or phrase (type 'exit' to quit)\n")

while True:

    user_input = input("Enter theme: ")

    if user_input.lower() == "exit":
        break

    result = generate_text(user_input, 15)

    print("\nGenerated Line:")
    print(result)
    print()
