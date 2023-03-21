import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from  tensorflow.keras.models import load_model
model = load_model('ait_chatbot_model.h5')
import json
import random
intents = json.loads(open('dataset.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def fnclean_sentence(query):
    query_in_words = nltk.word_tokenize(query)
    query_in_words = [lemmatizer.lemmatize(word.lower()) for word in query_in_words]
    return query_in_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def fnbow(query, words, show_details=True):
    # tokenize the pattern
    query_in_words = fnclean_sentence(query)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in query_in_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def fnpredict_class(query, model):
    # filter out predictions below a threshold
    p = fnbow(query, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def fngetResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            # result = (i['responses'])
            break
        elif(i['tag'] not in tag):
            result = 'not known'
        else:
            result = "You must ask the right questions"

    return result

def chatbot_response(msg):
    ints = fnpredict_class(msg, model)
    res = fngetResponse(ints, intents)
    return res
