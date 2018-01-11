from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


folder = './data/Weather/'

def merge():
  weather = {'air': pd.read_csv(folder + 'air_store_info_with_nearest_active_station.csv'),
             'hpg': pd.read_csv(folder + 'hpg_store_info_with_nearest_active_station.csv')}
  # print(weather['air'].head())
  # print(weather['hpg'].head())

  location = pd.read_csv('./data/location_info.csv', index_col=0)
  labels_air = ['air_store_id', 'station_id', 'station_latitude', 'station_longitude', 
    'station_vincenty', 'station_great_circle']
  labels_hpg = ['hpg_store_id', 'station_id', 'station_latitude', 'station_longitude', 
    'station_vincenty', 'station_great_circle']
  location = pd.merge(location, weather['air'][labels_air], on='air_store_id', how='left')
  location = pd.merge(location, weather['hpg'][labels_hpg], on='hpg_store_id', how='left')
  location = location.rename(columns={'station_id_x': 'air_station_id', 'station_latitude_x': 'air_station_latitude', 
    'station_longitude_x': 'air_station_longitude', 'station_vincenty_x': 'air_station_vincenty', 
    'station_great_circle_x':'air_station_great_circle', 'station_id_y': 'hpg_station_id',
    'station_latitude_y': 'hpg_station_latitude', 'station_longitude_y': 'hpg_station_longitude',
    'station_vincenty_y': 'hpg_station_vincenty', 'station_great_circle_y': 'hpg_station_great_circle'})
  location.to_csv('location_with_weather.csv')

def search_date_weather(station_id, date, folder='./data/Weather/data/'):
  file_name = folder + str(station_id) + '.csv'
  date_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
  tmp1 = pd.read_csv(file_name, index_col=0, date_parser=date_parser)
  wtinfo = tmp1[tmp1.index == date]
  return wtinfo

def merge_weather_to_main(visit_data, label='air'):
  store_id_label = label+ '_store_id'
  location = pd.read_csv('location_with_weather.csv', index_col=0)
  for i in range(len(visit_data)):
    tmp0 = visit_data.iloc[i]
    store_id = tmp0[store_id_label]
    date = tmp0['visit_date']
    try:
      tmp1 = location[location[store_id_label] == store_id]
      std_id = tmp1[label+'_station_id'].values[0]
      weather = search_date_weather(std_id, date)
      tmp2 = visit_data.iloc[i].append(weather.iloc[0])
      if 'result' not in locals():
        result = pd.DataFrame(tmp2).T
      else:
        result = result.append(tmp2, ignore_index=True)
    except:
      pass
    if (i+1) % 10000 == 0:
      print('  Now loading... %d th lines...' % (i+1))
  return result
  

def search_date_weather_from_raw():
  # RMK: There are some id have no std id...!!!
  air_visit = pd.read_csv('./data/air_visit_data.csv')
  air_visit['visit_date'] = pd.to_datetime(air_visit['visit_date'])
  tmp0 = air_visit.iloc[10020]
  air_id = tmp0['air_store_id']
  date = tmp0['visit_date']
  print(air_id)
  location = pd.read_csv('location_with_weather.csv', index_col=0)
  tmp1 = location[location['air_store_id'] == air_id]
  std_id = tmp1['air_station_id'].values[0]
  print(search_date_weather(std_id, date))

if __name__ == '__main__':
  main()
