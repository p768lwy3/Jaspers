""""
  API Key: https://developers.google.com/maps/documentation/geocoding/start

　　Time Series analysis:
    https://www.kaggle.com/fabiendaniel/recruit-restaurant-eda
    https://www.kaggle.com/thykhuely/recruit-time-series-interactive-eda
    https://www.kaggle.com/dongxu027/mean-mix-math-geo-harmonic-lb-0-493
    https://www.kaggle.com/cadong/weighted-average-on-four-kernels-lb-0-495 

  Weather:
    https://www.kaggle.com/huntermcgushion/weather-station-location-eda/notebook

"""
from gmaps import Geocoding
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

api_key = 'AIzaSyA8hsW9nZXvfAy10Qq5R_1MRbocIJnbvX4'
api = Geocoding(api_key=api_key)

folder = './input/'
path = {
  'air_visit': folder + 'air_visit_data.csv',
  'air_store': folder + 'air_store_info.csv',
  'air_reser': folder + 'air_reserve.csv',
  'hpg_store': folder + 'hpg_store_info.csv',
  'hpg_reser': folder + 'hpg_reserve.csv',
  'date_info': folder + 'date_info.csv',
  'samp_subm': folder + 'sample_submission.csv',
  'store_id': folder + 'store_id_relation.csv'
}


def read_data(path=path):
  # read all csv files
  air_visit = pd.read_csv(path['air_visit'])
  air_store = pd.read_csv(path['air_store'])
  air_reser = pd.read_csv(path['air_reser'])
  hpg_store = pd.read_csv(path['hpg_store'])
  hpg_reser = pd.read_csv(path['hpg_reser'])
  date_info = pd.read_csv(path['date_info'])
  samp_subm = pd.read_csv(path['samp_subm'])
  store_id = pd.read_csv(path['store_id'])

  # a. read reserve data:
  # i. preprocessing
  # 1. merge air and hpg by store_id
  air = pd.merge(air_reser, air_store, on='air_store_id')
  hpg = pd.merge(hpg_reser, hpg_store, on='hpg_store_id')
  air = pd.merge(air, store_id, how='left', on='air_store_id')
  hpg = pd.merge(hpg, store_id, how='left', on='hpg_store_id')
  result = pd.merge(air, hpg, how='outer')

  # 2. convert and split datetime into date and time
  result.visit_datetime = pd.to_datetime(result.visit_datetime)
  result.reserve_datetime = pd.to_datetime(result.reserve_datetime)
  result['visit_date'] = result.visit_datetime.dt.date
  result['visit_time'] = result.visit_datetime.dt.time
  result['reserve_date'] = result.reserve_datetime.dt.date
  result['reserve_time'] = result.reserve_datetime.dt.time

  # 3. megre date info to main
  date_info = date_info.rename({'calendar_date':'visit_date'}, axis='columns')
  date_info.visit_date = pd.to_datetime(date_info.visit_date).dt.date
  for column in ['day_of_week', 'holiday_flg']:
    result[column] = result['visit_date'].map(date_info.set_index('visit_date')[column].to_dict())

  # 4. megre extra information, like station location, weather etc...
  # 5. Saturday and Sunday flg?
  # 6. merge hpg and air genre name and area name?
  # 7. ???

  """
  # x. read visit data
  air_visit.visit_date = pd.to_datetime(data.air_visit.visit_date)

  # x. read sample submission
  samp_subm['visit_date'] = samp_subm['id'].map(lambda x: str(x).split('_')[2])
  samp_subm['visit_date'] = pd.todatetime(samp_subm['visit_date'])
  samp_subm['air_store_id'] = samp_subm['id'].map(lambda x: '_'.join(x.split('_')[:2]))
  samp_subm['dow'] = samp_subm['visit_date'].dt.dayofweek
  samp_subm['year'] = samp_subm['visit_date'].dt.year
  samp_subm['month'] = samp_subm['visit_date'].dt.month
  samp_subm['visit_date'] = samp_subm['visit_date'].dt.date
  """
  return result

## Getting Nearest Building with gmaps

def build_location_db(result, loc_path=folder+'location_info.csv'):
  location = result[['air_store_id', 'hpg_store_id', 'latitude', 'longitude']]
  for column in ['air_store_id', 'hpg_store_id']:
    location[column] = location[column].fillna(0)
  location_unique = location.groupby(['air_store_id', 'hpg_store_id']).mean()
  location_unique = location_unique.reset_index(level=['air_store_id', 'hpg_store_id'])
  location_unique.to_csv(loc_path)
  return loc_path

def search_nearest(latitude, longitude, target='JR station'):
  search = target + ' near ' + str(latitude) + ', ' + str(longitude)
  result_json = api.geocode(address=search)
  result_address = result_json[0]['formatted_address']
  result_lat = result_json[0]['geometry']['location']['lat']
  result_lng = result_json[0]['geometry']['location']['lng']
  return result_address, result_lat, result_lng

def nearest_dataframe(location, path=folder+'nearest_location.csv'):
  for i in range(len(location)):
    #try:
    lt, lg = location.loc[i, ['latitude', 'longitude']].values
    add, lat, lng = search_nearest(lt, lg)
    location.loc[i, 'nearest_jr_station_addr'] = add
    location.loc[i, 'nearest_jr_station_latitude'] = lat
    location.loc[i, 'nearest_jr_station_longitude'] = lng
    #except:
    #  pass
    if (i + 1)%10 == 0:
      print('  Here is the %d-th lines' % (i+1))
  location.to_csv(path)
  return path

def distance(location):
  location['distance'] = np.sqrt((location['latitude'] - location['nearest_jr_station_latitude'])**2
                         + (location['longitude'] - location['nearest_jr_station_longitude'])**2)
  return location

# logit distance: lambda x: 1/(1+np.exp(-x)) /tanh distance: np.tanh(1/x) <- to normalize the distance failed.
# using min max norm instead? x - min / max - min

"""
def same_area(rest_addr, station_addr):
  result = True
  for element in rest_addr:
    result = result and element in station_addr
  return result
"""
## KMeans 
def kmeans_loc_analysis(location, n_clusters=10, barplot=True):
  kmeans = KMeans(n_clusters=10, random_state=0).fit(location[['longitude','latitude']])
  location['cluster'] = kmeans.predict(location[['longitude','latitude']])

  if barplot == True:
    f,axa = plt.subplots(1, 2, figsize=(15,6))
    hist_clust = location.groupby(['cluster'], as_index=False).count()
    sns.barplot(x=hist_clust.cluster, y=hist_clust.air_store_id, ax=axa[0])
    sns.barplot(x=hist_clust.cluster, y=hist_clust.hpg_store_id, ax=axa[1])
    plt.show()

  return location

## Holiday analysis

## weather


