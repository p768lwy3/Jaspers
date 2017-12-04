import keras
import nltk
import numpy as np
import pandas as pd

from keras.constraints import non_neg
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

np.random.seed(7)
a2c = {'EAP': 0, 'HPL' : 1, 'MWS' : 2}
maxlen = 256
result_path = './sample_submission.csv'
stemmer = SnowballStemmer('english')
train_path = './train.csv'
test_path = './test.csv'
test_size = 0.2


def tokenizer(sent, stemmer=stemmer):
  sent = [w.lower() for w in nltk.word_tokenize(sent)]
  sent = [stemmer.stem(w) for w in sent]
  return sent

def read_train_file(path):
  raw = pd.read_csv(path)
  raw_text = raw.text.values
  print('  Loaded the file...')

  Xtk = []
  for sent in raw_text:
    Xtk.append(' '.join(tokenizer(sent)))
  print('  Tokenized X of the file...')
  tfvt = TfidfVectorizer(min_df=2, smooth_idf=True, sublinear_tf=True).fit(Xtk)
  tfvb = tfvt.vocabulary_
  Xvt = []
  for x in Xtk:
    sent = x.split()
    sent = [tfvb[w]+1 if tfvb.get(w) is not None else len(tfvb)+1 for w in sent]
    Xvt.append(np.array(sent))
  print('  Vectorized X of the file...')
  Xpad = pad_sequences(Xvt, maxlen=maxlen)
  print('  Padded X of the file...')
    
  y = np.array([a2c[a] for a in raw.author])
  y = to_categorical(y)
  print('  Vectorized y of the file...')

  x_train, x_test, y_train, y_test = train_test_split(Xpad, y, test_size=test_size)
  print('  Splited the file into two set...')
  return x_train, x_test, y_train, y_test, tfvb

def read_test_file(path, tfvb):
  raw = pd.read_csv(path)
  raw_text = raw.text.values
  print('  Loaded the file...')
  Xtk = []
  for sent in raw_text:
    Xtk.append(tokenizer(sent))
  print('  Tokenized X of the file...')
  Xvt = []
  for x in Xtk:
    sent = [tfvb[w]+1 if tfvb.get(w) is not None else len(tfvb)+1 for w in x]
    Xvt.append(np.array(sent))
  print('  Vectorized X of the file...')
  Xpad = pad_sequences(Xvt, maxlen=maxlen)
  print('  Padded X of the file...')
  return Xpad

def create_model(input_dim, embedding_dim=32, num_hidden_units=128, windows=4, 
                 num_recurret_layers=1, dropout_rate=0.2, output_dim=3, optimizer='adam'):
  model = Sequential()
  model.add(Embedding(input_dim=input_dim, output_dim=embedding_dim, embeddings_constraint=non_neg()))
  model.add(Conv1D(num_hidden_units, windows, activation='tanh'))
  model.add(Dropout(dropout_rate))
  for i in range(num_recurret_layers):
    model.add(Conv1D(num_hidden_units, windows, activation='relu'))
    model.add(Dropout(dropout_rate))
  model.add(GlobalMaxPooling1D())
  model.add(Dense(num_hidden_units, activation='relu'))
  model.add(Dropout(dropout_rate))
  model.add(Dense(output_dim, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  model.summary()
  return model

def main():
  x_train, x_test, y_train, y_test, tfvb = read_train_file(train_path)
  x_predict = read_test_file(test_path, tfvb)

  model = create_model(len(tfvb)+2)
  hist = model.fit(x_train, y_train, batch_size=16, validation_data=(x_test, y_test), epochs=10)
  y = model.predict(x_predict)

  result = pd.read_csv(result_path)
  for a, i in a2c.items():
    result[a] = y[:, i]
  result.to_csv(result_path, index=False)

if __name__ == '__main__':
  main()
