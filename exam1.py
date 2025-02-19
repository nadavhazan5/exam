import pandas as pd
import yaml

from preprocessor import Preprocessor
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
with open('product-reviews.yml', 'r') as f:
  reviews_df = pd.DataFrame(yaml.safe_load(f), columns=['review'])

# Preprocess for text anaylsis
preprocessor = Preprocessor()
reviews_df['preprocessed'] =  reviews_df['review'].apply(lambda review: preprocessor.preprocess(review))

# Get frequencies for topic extraction 
word_vectorizer = CountVectorizer(ngram_range=(1,3), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(reviews_df['preprocessed'])
frequencies = sum(sparse_matrix).toarray()[0]

# Display reviews' words by frequency
frequencies_matrix = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names_out(), columns=['frequency']).sort_values("frequency", ascending=False)
print(frequencies_matrix[:20])
