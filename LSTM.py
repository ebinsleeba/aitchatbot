import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the intent data from a JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Tokenize the text data and convert to sequences
tokenizer = Tokenizer()
all_patterns = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        all_patterns.append(pattern)
tokenizer.fit_on_texts(all_patterns)
max_sequence_length = max([len(seq) for seq in tokenizer.texts_to_sequences(all_patterns)])
num_words = len(tokenizer.word_index) + 1

sequences = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        sequence = tokenizer.texts_to_sequences([pattern])[0]
        sequences.append((sequence, intent['tag']))

# Create input and output data for the LSTM model
X, y = [], []
for sequence, intent_tag in sequences:
    padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length, padding='post')[0]
    X.append(padded_sequence)
    y.append(intents['tags'].index(intent_tag))

X = np.array(X)
y = np.array(y)

# Define the LSTM model
model = Sequential()
model.add(Embedding(num_words, 128, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(intents['tags']), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the LSTM model
model.fit(X, y, epochs=500, verbose=1)

# Use the trained model to classify user input and generate responses
def classify_intent(user_input):
    sequence = tokenizer.texts_to_sequences([user_input])[0]
    padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length, padding='post')
    prediction = model.predict(padded_sequence)[0]
    tag_index = np.argmax(prediction)
    tag = intents['tags'][tag_index]
    return tag

def generate_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = np.random.choice(intent['responses'])
            break
    return response

# Example usage
while True:
    user_input = input("You: ")
    intent_tag = classify_intent(user_input)
    response = generate_response(intent_tag)
    print("Bot: " + response)
