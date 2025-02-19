from bertopic import BERTopic
from gensim import corpora
from gensim.models.ldamodel import LdaModel

def extract_topics_lda(preprocessed_reviews):
  # Prepare data for LDA
  tokened_reviews = preprocessed_reviews.apply(lambda review: review.split(' '))
  # Load LDA Model
  dictionary = corpora.Dictionary(tokened_reviews)
  doc_term_matrix = tokened_reviews.apply(lambda review: dictionary.doc2bow(review))
  lda_model = LdaModel(corpus=doc_term_matrix, num_topics=3, id2word=dictionary, passes=10, random_state=45)
  # Display extracted topics
  for topic in lda_model.print_topics():
    print(topic)

def extract_topics_bert(preprocessed_reviews):
  # Load BERT Model
  topic_model = BERTopic()
  topics, probs = topic_model.fit_transform(preprocessed_reviews)
  # Display extracted topics
  print(topic_model.get_topic_info())
