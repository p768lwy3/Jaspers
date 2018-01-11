"""
Source:
  NavieBayesian: https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb/notebook
  LSTM: https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout-lb-0-048
"""

# import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Bidirectional, Dense, Dropout, Embedding, GlobalMaxPool1D, Input, LSTM
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

np.random.seed(7)

def read_file(folder='./input/'):
  # read from csv.
  train = pd.read_csv(folder + 'train.csv')
  test = pd.read_csv(folder + 'test.csv')
  subm = pd.read_csv(folder + 'sample_submission.csv')
  # build a none column and fill nan.
  train['comment_text'] = train['comment_text'].fillna('__na__')
  test['comment_text'] = test['comment_text'].fillna('__na__')
  return train, test, subm

def build_model(max_len, max_features, embed_size):
  inp = Input(shape=(max_len, ))
  x = Embedding(max_features, embed_size)(inp)
  x = Bidirectional(LSTM(64, return_sequences=True))(x)
  x = GlobalMaxPool1D()(x)
  x = Dropout(0.1)(x)
  x = Dense(64, activation='relu')(x)
  x = Dropout(0.1)(x)
  x = Dense(6, activation='sigmoid')(x)
  model = Model(inputs=inp, outputs=x)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(model.summary())
  return model

def main():
  # Parameters
  max_features = 5000
  max_len = 100
  embed_size = 32

  batch_size = 32
  epochs = 2
  file_path="weights_base.best.hdf5"

  # Build Training and Testing Data Set
  train, test, subm = read_file()
  list_sentences_train = train['comment_text'].values
  list_sentences_test = test['comment_text'].values
  y = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
  print('  Finish to Read Data.')

  tokenizer = Tokenizer(num_words=max_features)
  tokenizer.fit_on_texts(list(list_sentences_train))
  list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
  list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
  print('  Finish to Tokenize Sentence.')

  X_t = pad_sequences(list_tokenized_train, maxlen=max_len)
  X_te = pad_sequences(list_tokenized_test, maxlen=max_len)
  print('  Finish to Pad Sentence.')

  # Build the model
  model = build_model(max_len, max_features, embed_size)
  checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  earlyStopping = EarlyStopping(monitor="val_loss", mode="min", patience=20)
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, epsilon=0.0001, min_lr=0.001)
  callbacks_list = [checkpoint, earlyStopping, reduce_lr]
  print('  Finish to Build the model.')

  # Fit the model
  history = model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
  print('  Finish to Fit the model.')

  # Plot the model
  print('    Plotting accuracy and loss values......')
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model accuracy / loss')
  plt.ylabel('accuracy / loss')
  plt.xlabel('epoch')
  plt.legend(['train　accuracy', 'test accuracy', 'train loss', 'test loss'], loc='upper left')
  plt.show()
  print('  Finish to Plot the model.')
  
  # Save the prediction
  model.load_weights(file_path)
  y_test = model.predict(X_te)
  subm[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = y_test
  subm.to_csv("submission.csv", index=False)
  print('  Every Things are Finished.')

if __name__ == '__main__':
  main()
