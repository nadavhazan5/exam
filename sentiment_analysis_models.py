from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import f1_score, classification_report
from transformers import pipeline
from textblob import TextBlob

POSITIVE_SENTIMENT = 'POS'
NEGATIVE_SENTIMENT = 'NEG'
NEUTRAL_SENTIMENT = 'NEU'

HUGGINGFACE_SENTIMENTS = {
  "LABEL_0" : NEGATIVE_SENTIMENT,
  "LABEL_1" : NEUTRAL_SENTIMENT,
  "LABEL_2" : POSITIVE_SENTIMENT
}

def polarity_to_sentiment(polarity):
  # Convert numeric polarity score to sentiment code for analysis
  if polarity>0:
    return POSITIVE_SENTIMENT
  if polarity<0:
    return NEGATIVE_SENTIMENT
  return NEUTRAL_SENTIMENT

def evaluate_effectiveness(model_name, reviews_df):
  print(model_name + " effectiveness:")
  # Evaluate using accuracy
  correct = [x for x in range(len(reviews_df['pred'])) if reviews_df['pred'][x] == reviews_df['label'][x]]
  print("Accuracy: %{0}".format(100*(len(correct)/len(reviews_df['pred']))))
  # Evaluate using F1 Score
  print("F1 Score:", f1_score(reviews_df['label'], reviews_df['pred'], average="weighted"))

def sentiment_analysis_textblob(reviews_df):
  # Predict sentiment for each review
  reviews_df['pred'] = reviews_df['review'].apply(lambda review: polarity_to_sentiment(TextBlob(review).sentiment[0]))
  evaluate_effectiveness("Textblob", reviews_df)

def sentiment_analysis_huggingface(reviews_df):
  # Load sentiment analysis model
  sentiment_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
  # Predict sentiment for each review
  reviews_df["pred"] = reviews_df["review"].apply(lambda review: HUGGINGFACE_SENTIMENTS[sentiment_model(review)[0]["label"]])
  evaluate_effectiveness("HuggingFace", reviews_df)

def sentiment_analysis_nltk(reviews_df):
  # Load sentiment analysis model
  sid = SentimentIntensityAnalyzer()
  # Predict sentiment for each review
  reviews_df['pred'] = reviews_df['review'].apply(lambda review: polarity_to_sentiment(sid.polarity_scores(review)['compound']))
  evaluate_effectiveness("NLTK", reviews_df)
