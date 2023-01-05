import random
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

import pprint

import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

# class spam_util(obj):

    # def __init__(self):
    #     self.data_set = data_set
    #     self.message = messages_set
    #     self.wordlist = wordlist
    #     # self.document = document

def preprocess(document, stemmer, wordnet_lemmatizer, stem=True):
    # def preprocess(self, document, stem=True):
    'changes document to lower case, removes stopwords and lemmatizes/stems the remainder of the sentence'

    ## initialise the inbuilt Stemmer and the Lemmatizer
    # stemmer = PorterStemmer()
    # wordnet_lemmatizer = WordNetLemmatizer()

    # change sentence to lower case
    document = document.lower()

    # tokenize into words
    words = word_tokenize(document)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]

    if stem:
        words = [stemmer.stem(word) for word in words]
    else:
        words = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]

    # join words to make sentence
    document = " ".join(words)

    return document

## - creating a single list of all words in the entire dataset for feature list creation

def get_words_in_messages(messages):
    all_words = []
    for (message, label) in messages:
        all_words.extend(message)
    return all_words

## - creating a final feature list using an intuitive FreqDist, to eliminate all the duplicate words
## Note : we can use the Frequency Distribution of the entire dataset to calculate Tf-Idf scores like we did earlier.

def get_word_features(wordlist):
    #print(wordlist[:10])
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

## creating a LazyMap of feature presence for each of the 8K+ features with respect to each of the SMS messages
# def extract_features(document,word_features):
def extract_features(document):
    document_words = set(document)
    features = {}
    print(document_words)
    for word in get_word_features(document_words):
        features['contains(%s)' % word] = (word in document_words)
    return features

def pretty_printing(obj):
    pp = pprint.PrettyPrinter(indent=2)
    return pp.pprint(obj)

def preprocessing(data_set):
    ## initialise the inbuilt Stemmer and the Lemmatizer
    stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()

    ## - Performing the preprocessing steps on all messages
    messages_set = []
    for (message, label) in data_set:
        words_filtered = [e.lower() for e in preprocess(message, stem=False).split() if len(e) >= 3]
        messages_set.append((words_filtered, label))
    return messages_set

def storing_using_pickle(path_with_filename):
    #  Storing the classifier on disk for later usage
    import pickle
    f = open('path_with_filename', 'wb')
    pickle.dump(spamClassifier,f)
    print('Classifier stored at ', f.name)
    f.close()