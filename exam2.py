import pandas as pd
import yaml

from sentiment_analysis_models import sentiment_analysis_textblob, sentiment_analysis_huggingface, sentiment_analysis_nltk

# Load the data
with open('product-reviews.yml', 'r') as f:
  reviews_df = pd.DataFrame(yaml.safe_load(f), columns=['review'])

# Load preditermined labels
with open('product-reviews-labels.yml', 'r') as f:
  reviews_df['label'] = yaml.safe_load(f)

# Conduct Sentiment Analysis using different models
  sentiment_analysis_textblob(reviews_df)
  sentiment_analysis_huggingface(reviews_df)
  sentiment_analysis_nltk(reviews_df)