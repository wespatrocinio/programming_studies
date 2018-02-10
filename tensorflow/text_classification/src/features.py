import numpy as np
import nltk

from nltk.tokenize import RegexpTokenizer
from collections import Counter


nltk.download('stopwords')

def get_tokens(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    stopwords = nltk.corpus.stopwords.words("english")
    return [token for token in tokens if token not in stopwords]

def get_vocab(data):
    """ Split a raw text into a Counter indexed by word. """
    vocab = Counter()
    for text in data:
        for token in get_tokens(text):
            vocab[token] += 1
    return vocab

def get_len_vocab(text):
    return len(get_vocab(text))

def get_word_2_index(text):
    """ Generate an word:index dictionary. """
    word2index = {}
    for i,word in enumerate(get_vocab(text)):
        word2index[word.lower()] = i
    return word2index

def get_input_tensor(text):
    vocab = get_vocab(text)
    word2index = get_word_2_index(vocab)
    matrix = np.zeros(len(vocab), dtype=float)
    for word in text.split():
        matrix[word2index[word.lower()]] += 1
    return matrix

def get_output_tensor(category, n_classes):
    output = np.zeros(n_classes, dtype=float)
    output[category] = 1.0
    return output
