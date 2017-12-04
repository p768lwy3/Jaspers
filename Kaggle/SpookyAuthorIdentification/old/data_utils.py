"""
Preprocessing:
  1. Tokenization
  2. Throw away any words that occur too frequently or infrequently
  3. Stemming words
  4. Converting text into vector format

Function List:
  1. read_wiki2vec(path): 
      Read the txt-file contained pre-trained word2vec to dict.
      Ref Link: https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation
      
  2. simple_tokenizer(sentence, stemmer=stemmer, stopwords=stopwords):
      Tokenizing, Stemming, and Clearing Stopwords with a sentence.
      Return: List of words
      
  3. split_to_cross_valid(X, y, ratio=0.8):
      split the dataset of X and y into training set and cross validation set.
      Return: trainX, trainy, testX, testy
      
  4. bucketing(X_vector, y_train, padding=True, padding_from_head=False, var_len=200, buckets = [15, 25, 40, 75])
      seperate the dataset to different buckets by length.
      Return: dict with keys are bucket size and values are array of vectorized X or array of y
      Rmk: doing padding and using hstack with array to save the result is much slower compare with list.append()...

Rmk. 
  1. Tokenization with frequence/count to learn the use of words of author
      e.g. countvectorizer from sklearn (?)
  2. Using part of speech to learn the sentence structure of author
      Ref: http://www.nltk.org/book/ch05.html (5. Categorizing and Tagging Words)
  3. Sentiment Analysis to classify the type of sentence
      Ref: 
        https://marcobonzanini.com/2015/05/17/mining-twitter-data-with-python-part-6-sentiment-analysis-basics/
        http://pythonforengineers.com/build-a-sentiment-analysis-app-with-movie-reviews/
        http://zablo.net/blog/post/twitter-sentiment-analysis-python-scikit-word2vec-nltk-xgboost

Word/Paragraph/Text to Vectors:
  https://arxiv.org/abs/1507.07998
  http://analyzecore.com/2017/02/08/twitter-sentiment-analysis-doc2vec/
  http://building-babylon.net/2015/06/03/document-embedding-with-paragraph-vectors/
  http://cs.stanford.edu/~quocle/paragraph_vector.pdf
  https://deeplearning4j.org/doc2vec
  https://github.com/idio/wiki2vec/
  https://github.com/klb3713/sentence2vec/blob/master/utils.py
  https://stackoverflow.com/questions/17053459/how-to-transform-a-text-to-vector
  http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
  https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
  https://www.linguistics.rub.de/konvens16/pub/11_konvensproc.pdf
  https://www.quora.com/What-are-some-good-ways-to-represent-vectorize-words-phrases-and-sentences-in-deep-learning-applications
"""

# Import:
import nltk
import nltk.stem as stm
import numpy as np
import os.path
import pandas as pd

# Global Variables:
stopwords = nltk.corpus.stopwords.words('english')
stemmer = stm.SnowballStemmer("english")

def read_wiki2vec(path='./data/glove.6B.200d.txt'):
  d = {}
  f = open(path, 'r')
  for l in f:
    e = l.split()
    d[e[0]] = np.array([float(i) for i in e[1:]])
  return d

def simple_tokenizer(sentence, stemmer=stemmer, stopwords=stopwords):
  word_list = [word.lower() for w in nltk.word_tokenize(sentence)]
  if stemmer == stemmer:
    word_list = [stemmer.stem(w) for w in word_list]
  if stopwords == stopwords:
    word_list = [w for w in word_list if w not in stopwords]
  return word_list

def split_to_cross_valid(X, y, ratio=0.8):
  train_size = int(X.shape[0]*ratio)
  idx = np.random.permutation(X.shape[0])
  train_idx = idx[:train_size]
  test_idx = idx[train_size:]
  train_X = X[train_idx]; train_y = y[train_idx]
  test_X = X[test_idx]; test_y = y[test_idx]
  return train_X, train_y, test_X, test_y

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
Not Finished:
"""
'''
  try to convert words to part of spreech to analysis sentence structure 
  Ref: http://www.nltk.org/book/ch05.html

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
'''




"""
Extra (Quote from others):
  By using Lantent Dirichlet Allocation, print out Count Vectorizer result for visualisation.
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
