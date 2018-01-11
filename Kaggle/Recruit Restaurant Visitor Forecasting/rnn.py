## Import
import matplotlib.pyplot as plt, numpy as np, pandas as pd

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Bidirectional, concatenate, Dense, Dropout, Embedding, GlobalMaxPool1D, Input, LSTM
from keras.models import Model, Sequential
from keras.optimizers import Adam

np.random.seed(7)

folder = './input/'

def build_model(maxfeature, embedsize, features, windows):
  main_inp = Input(shape=(windows, ))
  x = Embedding(maxfeature, embedsize)(main_inp)
  x = Bidirectional(LSTM(64, return_sequences=True))(x)
  x = GlobalMaxPool1D()(x)
  x = Dropout(0.1)(x)
  x = Dense(64, activation='relu')(x)

  sub_inp = Input(shape=(features, ))
  x = concatenate([x, sub_inp])
  x = Dense(64, activation='relu')(x)
  x = Dropout(0.1)(x)
  x = Dense(64, activation='relu')(x)
  x = Dropout(0.1)(x)
  x = Dense(1, kernel_initializer='normal')(x)

  model = Model(inputs=[main_inp, sub_inp], outputs=x)
  model.compile(loss='mean_squared_error', optimizer='adam')

  print(model.summary())

  return model

def simple_lstm(windows):
  # structure
  model = Sequential()
  model.add(LSTM(8, input_shape=(windows, 1)))
  model.add(Dense(units=1))
  # compile
  optimizer = Adam(lr=0.01, epsilon=1e-08, decay=0.0)
  model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
  print(model.summary())
  return model

def read_data():
  path = {
    'air_visit': folder + 'air_visit_data.csv',
    'air_store': folder + 'air_store_info.csv',
    'air_reser': folder + 'air_reserve.csv',
    'hpg_store': folder + 'hpg_store_info.csv',
    'hpg_reser': folder + 'hpg_reserve.csv',
    'date_info': folder + 'date_info.csv',
    'samp_subm': folder + 'sample_submission.csv',
    'store_id': folder + 'store_id_relation.csv',
    'jr_info': folder + 'jr_location.csv'
  }
  air_visit = pd.read_csv(path['air_visit'])
  air_visit['visit_date'] = pd.to_datetime(air_visit['visit_date'])
  air_store = pd.read_csv(path['air_store'])
  #air_reser = pd.read_csv(path['air_reser'])
  #hpg_store = pd.read_csv(path['hpg_store'])
  #hpg_reser = pd.read_csv(path['hpg_reser'])
  date_info = pd.read_csv(path['date_info'])
  date_info = date_info.rename(columns={'calendar_date':'visit_date'})
  date_info['visit_date'] = pd.to_datetime(date_info['visit_date'])
  #jr_info = pd.read_csv(path['jr_info'])
  #jr_info = jr_info[['air_store_id', 'nearest_jr_station_latitude', 'nearest_jr_station_longitude']]
  #jr_info = jr_info.rename(columns={'nearest_jr_station_latitude':'jr_lat', 'nearest_jr_station_longitude':'jr_lng'})
  
  #samp_subm = pd.read_csv(path['samp_subm'])
  #store_id = pd.read_csv(path['store_id'])

  # Rescaler
  #scaler = MinMaxScaler().fit(adjclose)
  #data = scaler.transform(adjclose) # data shape = (len(data), 1)

  for column in ['day_of_week', 'holiday_flg']:
    air_visit[column] = air_visit['visit_date'].map(date_info.set_index('visit_date')[column].to_dict())

  #for column in ['jr_lat', 'jr_lng']:
  #  air_visit[column] = air_visit['air_store_id'].map(jr_info.set_index('air_store_id')[column].to_dict())

  return air_visit, air_store

def main():
  air_visit, air_store = read_data()

  # Input
  spt = {}
  gups = air_visit.groupby('air_store_id')
  for air_store_id, gup in gups:
    spt[air_store_id] = gup
  print('Finish splitting...')

  windows = 7
  main_inp, sub_inp, y = np.array([]), np.array([]), np.array([])
  for k, v in spt.items():
    vts = v.visitors.values
    pvt = np.array([vts[i:i+windows] for i in range(len(vts)-windows)])
    pvt = np.concatenate((np.full([windows, windows], np.nan), pvt))
    pvt = pd.DataFrame(pvt, columns=['pv1', 'pv2', 'pv3', 'pv4', 'pv5', 'pv6', 'pv7']).reset_index(drop=True)
    v = v.reset_index(drop=True)
    v = pd.concat([v, pvt], axis=1).dropna()
    # main input, time series slicing by windows
    main_inp = np.append(main_inp, v[['pv1', 'pv2', 'pv3', 'pv4', 'pv5', 'pv6', 'pv7']].values)

    # sub input
    v_year = v.visit_date.dt.year.values
    v_month = v.visit_date.dt.month.values
    v_day = v.visit_date.dt.day.values
    v_dow = v.visit_date.dt.dayofweek.values
    v_hf = v['holiday_flg'].values

    air_store_id = v.air_store_id.values[0]
    lat = air_store[air_store['air_store_id']==air_store_id]['latitude'].values[0]
    lng = air_store[air_store['air_store_id']==air_store_id]['longitude'].values[0]
    # air_area_name, air_genre_name, <= they need embedding
    sub_tmp = np.vstack((v_year, v_month, v_day, v_dow, v_hf, np.array([lat]*len(v)), np.array([lng]*len(v)))).T
    sub_inp = np.append(sub_inp, sub_tmp)
    # output
    y = np.append(y, v.visitors.values)
  # Reshape
  main_inp = main_inp.reshape(int(main_inp.shape[0]/windows), windows)
  sub_inp = sub_inp.reshape(int(sub_inp.shape[0]/7), 7)
  print('Finish reshaping...')

  # Model...
  maxfeature = 1000
  embedsize = 16
  features = sub_inp.shape[1]
  model = build_model(maxfeature, embedsize, features, windows)
  print('Finish building model and start to fit......')

  # Fit
  file_path = "weights_base.best.hdf5"
  checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  earlyStopping = EarlyStopping(monitor="val_loss", mode="min", patience=20)
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, epsilon=0.0001, min_lr=0.0001)
  callbacks = [checkpoint, earlyStopping, reduce_lr]

  history = model.fit([main_inp, sub_inp], y, epochs=20, batch_size=128, validation_split=0.1, callbacks=callbacks)

  train_error = model.evaluate([main_inp, sub_inp], y, verbose=0)
  print('Train Error: ', train_error)
  
  print('Plotting loss values......')
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model / loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train loss', 'test loss'], loc='upper left')
  plt.show()
  print('  Finish to Plot the model.')

def read_submission(path='./input/sample_submission.csv'):
  csv = pd.read_csv(path)
  csv['air_store_id'] = csv['id'].map(lambda x: '_'.join(str(x).split('_')[:2]))
  csv['visit_date'] = csv['id'].map(lambda x: str(x).split('_')[2])
  csv['visit_date'] = pd.to_datetime(csv['visit_date'])
  csv['year'] = csv['visit_date'].dt.year
  csv['month'] = csv['visit_date'].dt.month
  csv['day'] = csv['visit_date'].dt.day
  csv['dayofweek'] = csv['visit_date'].dt.dayofweek
  csv = csv.drop('id', axis=1)

  date_info = pd.read_csv('./input/date_info.csv')
  date_info = date_info.rename(columns={'calendar_date':'visit_date'})
  date_info['visit_date'] = pd.to_datetime(date_info['visit_date'])
  csv['holiday_flg'] = csv['visit_date'].map(date_info.set_index('visit_date')['holiday_flg'].to_dict())

  return csv

def prediction(windows=7):
  # read data
  air_visit = pd.read_csv('./input/air_visit_data.csv')
  air_visit['visit_date'] = pd.to_datetime(air_visit['visit_date'])
  air_store = pd.read_csv('./input/air_store_info.csv')
  samp_sumb = read_submission()
  print('  Read data...')

  samp_spt, spt = {}, {}
  samp_gups = samp_sumb.groupby('air_store_id')
  gups = air_visit.groupby('air_store_id')
  for air_store_id, gup in samp_gups:
    samp_spt[air_store_id] = gup
  for air_store_id, gup in gups:
    spt[air_store_id] = gup
  print('  Grouped data...')

  model = build_model(1000, 16, 7, 7)
  model.load_weights("weights_base.best.hdf5")
  print('  Loaded Model...')

  visitors = []
  print('  Started to predict...')
  for key in samp_spt.keys():
    #print('    The air store id is %s' % key)
    past = spt[key]
    last_windows = list(past.visitors.values[-windows:])
    v_year = samp_spt[key].visit_date.dt.year.values
    v_month = samp_spt[key].visit_date.dt.month.values
    v_day = samp_spt[key].visit_date.dt.day.values
    v_dow = samp_spt[key].visit_date.dt.dayofweek.values
    v_hf = samp_spt[key]['holiday_flg'].values

    lat = air_store[air_store['air_store_id']==key]['latitude'].values[0]
    lng = air_store[air_store['air_store_id']==key]['longitude'].values[0]

    for i in range(len(samp_spt[key])):
      main_inp = np.array(last_windows).reshape(-1, len(last_windows))
      sub_inp = np.array([v_year[i], v_month[i], v_day[i], v_dow[i], v_hf[i], lat, lng])
      sub_inp = sub_inp.reshape(-1, len(sub_inp))
      y_pred = model.predict([main_inp, sub_inp])
      visitors.append(y_pred)
      samp_spt[key]['visitors'][i] = y_pred
      last_windows.pop(0)
      last_windows.append(y_pred)
  return visitors

  """
  sumb = pd.read_csv('./input/air_visit_data.csv')
  sumb['visitors'] = visitors
  sumb.visitors = sumb.visitors.map(lambda x: x[0, 0])
  sumb.to_csv('submission.csv', index=False)    
  """


if __name__ == '__main__':
  prediction()
