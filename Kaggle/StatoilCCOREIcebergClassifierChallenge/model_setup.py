"""
# Source:
https://www.kaggle.com/cbryant/keras-cnn-statoil-iceberg-lb-0-1995-now-0-1516
https://zhuanlan.zhihu.com/p/30756859
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(11550)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

def get_scaled_imgs(df):
  imgs = []
  for i, row in df.iterrows():
    band_1 = np.array(row['band_1']).reshape(75,75)
    band_2 = np.array(row['band_2']).reshape(75,75)
    band_3 = band_1 + band_2

    a = (band_1 - band_1.mean())/(band_1.max()-band_1.min()) # band_1.std() seems worse
    b = (band_2 - band_1.mean())/(band_2.max()-band_2.min())
    c = (band_3 - band_3.mean())/(band_3.max()-band_3.min())

    imgs.append(np.dstack((a, b, c)))

  return np.array(imgs)

def read_data():
  df_train = pd.read_json('./data/train.json')
  Xtrain = get_scaled_imgs(df_train)
  Ytrain = np.array(df_train['is_iceberg'])

  df_train.inc_angle = df_train.inc_angle.replace('na', 0)
  idx_tr = np.where(df_train.inc_angle > 0)

  Xtrain = Xtrain[idx_tr[0], ...]
  Ytrain = Ytrain[idx_tr[0]]

  return Xtrain, Ytrain

def model_builder():
  model = Sequential()

  # CNN 1
  model.add(Conv2D(128, kernel_size=(4,4), activation='relu', input_shape=(75,75,3)))
  model.add(MaxPooling2D(pool_size=(2,2))) #, strides=(2,2)
  #model.add(Dropout(0.05))

  # CNN 2
  model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2))) #, strides=(2,2)
  #model.add(Dropout(0.05))
  
  # CNN 3
  model.add(Conv2D(256, kernel_size=(2,2), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2))) #, strides=(2,2)
  #model.add(Dropout(0.05))

  # CNN 4
  model.add(Conv2D(128, kernel_size=(2,2), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2))) #, strides=(2,2)
  #model.add(Dropout(0.05))

  # Flatten
  model.add(Flatten())

  # Dense 1
  model.add(Dense(512, activation='relu'))
  #model.add(Dropout(0.05))

  # Dense 2
  model.add(Dense(256, activation='relu'))
  #model.add(Dropout(0.05))

  # Dense 3
  model.add(Dense(128, activation='relu'))
  #model.add(Dropout(0.05))

  # Output
  model.add(Dense(1, activation='sigmoid'))

  optimizer = Adam(lr=0.0005, decay=0.0)
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  return model

def main():
  print('    Start to read data......')
  Xtrain, Ytrain = read_data()
  Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain, Ytrain, test_size=0.1, random_state=42)
  print('    Finish to read data.')

  # Image Generator
  print('    Start to build Image Data Generator......')
  imggen = ImageDataGenerator(
    width_shift_range=0.0025,
    height_shift_range=0.0025,
    horizontal_flip=True,
    vertical_flip=True)
  imggen.fit(Xtrain)
  print('    Finish to build Image Data Generator.')

  # Call Back function to save the model
  print('    Start to set Call Back Function......')
  earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
  mdlfp = './model/weights.best.mdl_wts.hdf5'
  mcp_save = ModelCheckpoint(mdlfp, save_best_only=True, monitor='val_loss', mode='min')
  reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
  callbacks = [earlyStopping, mcp_save, reduce_lr_loss]
  print('    Finish to set Call Back Function.')

  # Fit the model
  print('    Start to build the Neural Network...')
  model = model_builder()
  print(model.summary())
  batch_size = 8
  epochs = 100
  history = model.fit_generator(imggen.flow(Xtrain, Ytrain, batch_size=batch_size), steps_per_epoch=len(Xtrain)/32, epochs=epochs, verbose=1, callbacks=callbacks, validation_data=(Xtest, Ytest))
  print('    Finish to build the NN.')

  # Save model
  model_json = model.to_json()
  with open('./model/model.json', 'w') as json_file:
    json_file.write(model_json)

  # Plot accuracy / loss
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

  return Xtrain, Xtest, Ytrain, Ytest, model, mdlfp, history
  

if __name__ == '__main__':
  Xtrain, Xtest, Ytrain, Ytest, model, mdlfp, history = main()

  # load the best weights and check the score with the training and validation data.
  model.load_weights(filepath = mdlfp)

  # evaluate
  score_train = model.evaluate(Xtrain, Ytrain, verbose=1)
  score_valid = model.evaluate(Xtest, Ytest, verbose=1)
  print('    Train score: ', score_train[0], ', Train accuracy: ', score_train[1])
  print('    Validation score: ', score_valid[0], ', Validation accuracy: ', score_valid[1])

  # submission
  print('    Computing Results for submission......')
  df_test = pd.read_json('./data/test.json')
  df_test.inc_angle = df_test.inc_angle.replace('na', 0)
  test = (get_scaled_imgs(df_test))
  test_pred = model.predict(test)

  submission = pd.DataFrame({'id': df_test['id'], 'is_iceberg': test_pred.reshape((test_pred.shape[0]))})
  print(submission.head(10))
  submission.to_csv('submission.csv', index=False)
  print('    Saved submission.')
  print('  Finished......')
