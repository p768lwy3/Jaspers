"""
  The File for runing neural network model
"""

# Import:
import json
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten, GlobalAveragePooling1D, GlobalMaxPooling1D, LSTM, TimeDistributed
from keras.optimizers import SGD

import data_utils

def create_model(var_len=200, num_recurrent_units=3):
  nn = Sequential()
  # Convolution1D
  nn.add(Conv1D(256, input_shape=(None, 200), 
    batch_input_shape=(200, None, 200), kernel_size=32, padding='causal', activation='tanh'))
  nn.add(Dropout(0.2))
  for i in range(num_recurrent_units):
    dim_size = int(128/2**(i+1))
    kernel_size = int(32/2**(i+1))
    nn.add(Conv1D(dim_size, kernel_size=kernel_size, padding='causal', activation='tanh'))
    nn.add(Dropout(0.2))
  # nn.add(Conv1D(64, kernel_size=16, padding='causal', activation='relu'))
  # MaxPooling/GlobalAveragePooling
  # from 3D to 2D or GlobalAveragePooling1D, since Flatten() is only dealing with fixed input shape
  # Ref: https://github.com/fchollet/keras/issues/1920
  nn.add(GlobalMaxPooling1D()) 
  nn.add(Dropout(0.2))
  nn.add(Dense(8, activation='tanh'))
  nn.add(Dropout(0.2))
  nn.add(Dense(3, activation='softmax'))
  
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  nn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
  print(nn.summary())
  return nn

def main(PLOT=True, SAVE=False, ALLDATA=True, batch_size=200):
  # Create model
  model = create_model()
  print('  Model is Created......')

  # Read file
  X = np.load('./data/X_train_15.npy')
  y = np.load('./data/y_train_15.npy')

  X0, y0, X1, y1 = data_utils.split_to_cross_valid(X, y, ratio=0.8)
  X0 = np.delete(X0, range(3200, 3245), 0)
  y0 = np.delete(y0, range(3200, 3245), 0)
  X1 = np.delete(X1, range(800, 812), 0)
  y1 = np.delete(y1, range(800, 812), 0)
  y0 = keras.utils.to_categorical(y0, num_classes=3)
  y1 = keras.utils.to_categorical(y1, num_classes=3)
  print('  Data is Read......')

  hist = model.fit(X0, y0, nb_epoch=20, batch_size=batch_size, validation_data=(X1, y1))

  if ALLDATA == True:
    for i in ['25', '45', '75']:
      print('  Now is training length %s dataset...' % i)
      Xpath = './data/X_train_{0}.npy'.format(i)
      ypath = './data/y_train_{0}.npy'.format(i)
      X = np.load(Xpath)
      y = np.load(ypath)
      X0, y0, X1, y1 = data_utils.split_to_cross_valid(X, y, ratio=0.8)
      train0 = (len(X0)//batch_size)*batch_size; train1 = len(X0)
      test0 = (len(X1)//batch_size)*batch_size; test1 = len(X1)
      X0 = np.delete(X0, range(train0, train1), 0)
      y0 = np.delete(y0, range(train0, train1), 0)
      X1 = np.delete(X1, range(test0, test1), 0)
      y1 = np.delete(y1, range(test0, test1), 0)
      y0 = keras.utils.to_categorical(y0, num_classes=3)
      y1 = keras.utils.to_categorical(y1, num_classes=3)
      print('  Data is Read......')
      hist = model.fit(X0, y0, nb_epoch=20, batch_size=batch_size, validation_data=(X1, y1))

  # plot model:
  dtime = datetime.now()
  dstring = ''.join([str(dtime.year), str(dtime.month), 
    str(dtime.day), str(dtime.hour), str(dtime.minute), str(dtime.second)])
  if PLOT == True:
    history_dict = hist.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)

    print('  Plotting Loss Function...')
    plt.plot(epochs, loss_values, 'bo')
    plt.plot(epochs, val_loss_values, 'b+')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    lpath = './pic/lossfn_' + dstring + '.png'
    plt.savefig(lpath)
    plt.show()
    plt.clf()

    print('  Plotting Accuracy Function...')
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    plt.plot(epochs, acc_values, 'bo')
    plt.plot(epochs, val_acc_values, 'b+')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    apath = './pic/accuracyfn_' + dstring + '.png'
    plt.savefig(apath)
    plt.show()

  # save model:
  if SAVE == True:
    jpath = './data/model_' + dstring + '.json'
    mpath = './data/model_' + dstring + '.h5'
    model_json = model.to_json()
    with open(jpath, 'w') as jfile:
      jfile.write(model_json)
    model.save_weights(mpath)
    print('  Saved Model to disk...')

if __name__ == '__main__':
  main()
