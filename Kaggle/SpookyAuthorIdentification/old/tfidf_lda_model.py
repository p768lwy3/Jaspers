"""
  Building LDA or TF-Idf array to analysis, however the accuracy is quite low...
"""

## Import:
import keras
import numpy as np
import pandas as pd

from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import data_utils

n_features = 4096
no_of_category = 3
num_hidden_units = 512
num_recurrent_units = 3
batch_size = 256
epochs = 50
validation_split = 0.4

def create_model(input_dim=n_features, no_of_category=no_of_category, 
                 num_hidden_units=num_hidden_units, num_recurrent_units=num_recurrent_units):
  model = Sequential()
  model.add(Dense(num_hidden_units, input_dim=input_dim, activation='relu'))
  for curr_units in range(num_recurrent_units):
    model.add(Dense(num_hidden_units, activation='relu'))
    model.add(Dropout(0.2))
  model.add(Dense(no_of_category, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  print(model.summary())
  return model
  
def print_top_words(model, feature_names, n_top_words=10):
  for topic_ix, topic in enumerate(model.components_):
    message = "Topic #%d: " % topic_ix
    message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]])
    print(message)
    print()

def main(method='tfidf'):
  ds_raw = pd.read_csv('train.csv')
  X_raw = ds_raw.text.values
  y_raw = ds_raw.author.values
  
  for i in range(len(raw_X[i])):
    raw_X[i] = raw_X[i].lower()
  
  if method != 'tfidf':
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
    X_vector = tf_vectorizer.fit_transform(raw_X)
    if method == 'tf':
      X_vector = X_vector.toarray() #?
    elif method == 'lda':
      lda = LatentDirichletAllocation(n_topics=n_features, max_iter=10, learning_method='online',
                                      learning_offset=50, random_state=0).fit(X_vector)
      X_vector = lda.transform(X_vector)
      """
        print out LDA to see:
        tf_feature_names = tf_vectorizer.get_feature_names()
        print_top_words(lda, tf_feature_names, n_top_words=10)
      """
      
    
  elif method == 'tfidf':
    X_vector = data_utils.tokenizer_tfidf(X_raw) # add back the features variable.
  
  y_dict = {list(set(y_raw))[i]:i for i in range(len(set(y_raw)))}
  for i in range(len(y_raw)):
    y_raw[i] = y_dict[y_raw[i]]
  y_category = keras.utils.to_categorical(y_raw, num_classes=no_of_category)
  
  model = create_model()
  history = model.fit(X_vector, y_category, batch_size=batch_size, epochs = epochs, validation_split=validation_split)
  
  
