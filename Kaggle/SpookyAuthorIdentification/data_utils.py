# Import:
import nltk
import nltk.stem as stm
import numpy as np
import os.path
import pandas as pd

# Global Variables:
stopwords = nltk.corpus.stopwords.words('english')
stemmer = stm.SnowballStemmer("english")

def tokenizer(sentence):
  """ tokenize and do some preprocessing, like stemming and lowering, """
  return [stemmer.stem(word.lower()) for word in nltk.word_tokenize(sentence) if word.lower() not in stopwords]
  
def read_wiki2vec(path='./data/glove.6B.200d.txt'):
  """
    Using the pre trained vectorizer to vectorize words' tokens, 
    Link: https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation
  """
  d = {}
  f = open(path, 'r')
  for l in f:
    e = l.split()
    d[e[0]] = np.array([float(i) for i in e[1:]])
  return d

"""
  Tokenization with occuration to learn the use-of-word of author
     Ref: http://scikit-learn.org/stable/modules/feature_extraction.html
     Rmk: is this useful for classification with around 20000 samples?
"""
def count_vector(sent, max_df=0.95, min_df=2):
  tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english')
  tf = tf_vectorizer.fit_transform(sent).toarray() # shape = (19xxx, 15xxx)
  return tf

"""
  Bucketing, 
  rmk: doing padding and using hstack with array to save the result is much slower compare with list.append()...
"""
def bucketing(X_vector, y_train, padding=True, padding_from_head=False, var_len=200, buckets = [15, 25, 40, 75]):
  dx = {}; dy = {}
  counter = 0
  for i in range(len(X_vector)):
    s = X_vector[i]
    y = y_train[i]
    for b in buckets:
      if len(s) <= b:
        if padding == True:
          zeros = np.zeros([b-len(s), var_len])
          s = np.vstack([zeros, s]) if padding_from_head == True else np.vstack([s, zeros])
        if b not in d:
          dx[b] = np.array([s])
          dy[b] = np.array([y])
        else:
          dx[b] = np.append(dx[b], [s], axis=0)
          dy[b] = np.append(dy[b], [y], axis=0)
        break
    counter += 1
    if counter % 1000 == 0:
      print('  Now is the %d-th lines...' % counter)
  return d

"""
  Extra: By using Lantent Dirichlet Allocation, plot out Count Vectorizer result for visualisation,
"""
def lda_top_words(word_arr, n_topics=3, max_iter=5, learning_method='online', 
  learning_offset=50., random_state=0, n_top_words=20):
  
  tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
  tf = tf_vectorizer.fit_transform(word_arr)
  lda = LantentDirichletAllocation(n_topics=n_topics, max_iter=max_iter,
          learning_method=learning_method, learning_offset=learning_offset, random_state=random_state)
  lda.fit(tf)
  print("\nTopics in LDA model: ")
  tf_feature_names = tf_vectorizer.get_feature_names()
  print_top_words(lda, tf_feature_names, n_top_words)

# Define helper function to print top words
def print_top_words(model, feature_names, n_top_words):
  for index, topic in enumerate(model.components_):
    message = "\nTopic #{}:".format(index)
    message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
    print(message)
    print("="*70)

"""
'''
  try to convert words to part of spreech to analysis sentence structure 
  Ref: http://www.nltk.org/book/ch05.html
'''
def tokenize_with_pos(dataset):
  map_dict = {}
  dataset_pos = []
  for sent in dataset:
    sent_tk = nltk.tokenize.word_tokenize(sent) 
    sent_pos = nltk.pos_tag(sent_tk)
    p0 = []
    for pos in sent_pos:
      if pos[1] not in map_dict:
        map_dict(pos[1]) = len(map_dict)
      p0.append(map_dict[pos[1]])
    dataset_pos.append(np.array(p0))
  return np.array(dataset_pos)
"""

