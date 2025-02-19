import pandas as pd
import re
import string

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

class Preprocessor():
  def __init__(self):
    self.stop_words = set(stopwords.words('english')) 
    self.stemmer = PorterStemmer()
    self.lemmatizer = WordNetLemmatizer()

  def preprocess(self, review):
    review = review.lower()  # Lowercase
    review = review.translate(str.maketrans('', '', string.punctuation)) # Noise removal
    review = re.sub(r'\d+', '', review)  # Remove numbers
    tokens = word_tokenize(review)  # Tokenization
    tokens = [word for word in tokens if word not in self.stop_words]  # Remove stopwords
    tokens = [self.stemmer.stem(token) for token in tokens] # Stemming
    tokens = [self.lemmatizer.lemmatize(token) for token in tokens] # Lemmatization
    return " ".join(tokens)
