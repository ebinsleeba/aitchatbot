import json
import numpy as np
from gensim.models import Word2Vec

# Load the intent data from a JSON file
with open('dataset.json', 'r') as f:
    intents = json.load(f)

# Train a Word2Vec model on the text data
sentences = []
for intent in intents['intents']:
    sentences.extend([pattern.split() for pattern in intent['patterns']])
model = Word2Vec(sentences, min_count=1, vector_size=100)

#  old numpy 1.19.5
#  new numpy 1.24.5

# Use the trained Word2Vec model to classify user input and generate responses
def classify_intent(user_input):
    user_input_tokens = user_input.split()
    vectors = [model.wv[token] for token in user_input_tokens if token in model.wv]
    if len(vectors) > 0:
        vector_sum = np.sum(vectors, axis=0)
        vector_mean = vector_sum / len(vectors)
        similarity_scores = model.wv.cosine_similarities(vector_mean, model.wv.vectors)
        tag_index = np.argmax(similarity_scores)
        tag = intents['intents'][tag_index]['tag']
    else:
        tag = None
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
    if intent_tag:
        response = generate_response(intent_tag)
    else:
        response = "I'm sorry, I don't understand."
    print("Bot: " + response)
