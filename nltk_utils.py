"""
Created on Mon Aug  2 10:03:35 2021

@author: mazen saleh
"""
import numpy as np # using for numrical opertaion 
import nltk     #nltk its using to NLP preprocessing like: tokenize and stemming and bag of word
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize (sentence): 
     """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
     return nltk.word_tokenize(sentence)

def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())
 
def bag_of_word(tokenized_sentence,all_words): 
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    tokenized_sentence=[stem(w) for w in tokenized_sentence]  # stem each word

    bag=np.zeros(len(all_words), dtype=np.float32)  # initialize bag with 0 for each word
    for index, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[index]=1.0
    return bag          
