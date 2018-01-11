'''
  Data Visualization of Japanese Journey
  ## Source: https://www.kaggle.com/asindico/a-japanese-journey

https://www.kaggle.com/headsortails/be-my-guest-recruit-restaurant-eda
https://www.kaggle.com/captcalculator/a-very-extensive-recruit-exploratory-analysis
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


def read_data(folder='./data/'):
  airres = pd.read_csv(folder + 'air_reserve.csv')
  airstore = pd.read_csv(folder + 'air_store_info.csv')
  hpgres = pd.read_csv(folder + 'hpg_reserve.csv')
  hpgstore = pd.read_csv(folder + 'hpg_store_info.csv')
  airvisit = pd.read_csv(folder + 'air_visit_data.csv')

  air = pd.merge(airres, airstore, on='air_store_id')
  hpg = pd.merge(hpgres, hpgstore, on='hpg_store_id')
  rel = pd.read_csv(folder + 'store_id_relation.csv')
  airrel = pd.merge(air, rel, how='left', on='air_store_id')
  hpgrel = pd.merge(hpg, rel, how='left', on='hpg_store_id')
  full = pd.merge(airrel, hpgrel, how='outer')

  return air, hpg, airrel, hpgrel, full

def kmeans(full, n_clusters=10, barplot=True):
  kmeans = KMeans(n_clusters=10, random_state=0).fit(full[['longitude','latitude']])
  full['cluster'] = kmeans.predict(full[['longitude','latitude']])

  if barplot == True:
    f,axa = plt.subplots(1, 2, figsize=(15,6))
    hist_clust = full.groupby(['cluster'], as_index=False).count()
    sns.barplot(x=hist_clust.cluster, y=hist_clust.air_store_id, ax=axa[0])
    sns.barplot(x=hist_clust.cluster, y=hist_clust.hpg_store_id, ax=axa[1])
    plt.show()

  return full

def barplot(air, hpg):
  f,ax = plt.subplots(1, 1, figsize=(15,6))
  airhist = air.groupby(['air_store_id'], as_index=False).count()
  sns.distplot(airhist.visit_datetime)
  hpghist = hpg.groupby(['hpg_store_id'], as_index=False).count()
  sns.distplot(hpghist.visit_datetime)
  plt.show()
  #sns.barplot(x=airhist.air_store_id, y=airhist.visit_datetime)

def cousine_genres(full):
  air_genre = full.loc[full.air_genre_name.isnull()==False].groupby(['cluster','air_genre_name'],as_index=False).count()
  hpg_genre = full.loc[full.hpg_genre_name.isnull()==False].groupby(['cluster','hpg_genre_name'],as_index=False).count()
  genres = air.air_genre_name.unique()

  #i = 0
  f,axa= plt.subplots(2,1,figsize=(15,36))
  hm = []
  for i in range(10):
    genres_count = [ air_genre.loc[air_genre.cluster==i].loc[air_genre.air_genre_name==name]['air_store_id'].values[0] if name in air_genre.loc[air_genre.cluster==i].air_genre_name.values else 0 for name in genres] 
    hm.append(genres_count)
  hm = pd.DataFrame(hm,columns=genres,)
  sns.heatmap(hm.transpose(),cmap="YlGnBu",ax=axa[0])
  genres = hpg.hpg_genre_name.unique()
  hm = []
  for i in range(10):
    genres_count = [ hpg_genre.loc[hpg_genre.cluster==i].loc[hpg_genre.hpg_genre_name==name]['hpg_store_id'].values[0] if name in hpg_genre.loc[hpg_genre.cluster==i].hpg_genre_name.values else 0 for name in genres] 
    hm.append(genres_count)
  hm = pd.DataFrame(hm,columns=genres,)
  sns.heatmap(hm.transpose(),cmap="YlGnBu",ax=axa[1])
  plt.show()

def japanese_holidays(full, folder='./data/'):
  dates = pd.read_csv(folder + 'date_info.csv')
  dates.loc[dates.holiday_flg==1].loc[(dates.day_of_week !='Saturday')].loc[dates.day_of_week !='Sunday']

  vdt = pd.to_datetime(full.visit_datetime)
  rdt = pd.to_datetime(full.reserve_datetime)
  full['vd']=vdt.dt.date
  full['vt']=vdt.dt.time
  full['rd']=rdt.dt.date
  full['rt']=rdt.dt.time

  dts = pd.to_datetime(dates.calendar_date)
  days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
  dates['calendar_date'] = pd.to_datetime(dates['calendar_date']).dt.date
  dates['dy'] = dts.dt.dayofyear
  dates['dw'] = [days.index(dw) for dw in dates.day_of_week]
  dates.head()

  nf = pd.merge(full,dates[['calendar_date','holiday_flg']],how='left',left_on='vd',right_on='calendar_date')
  nf = nf.rename(index = str, columns = {'holiday_flg':'visit_holiday'})
  nf = nf.drop(['calendar_date'],axis=1)

  nf = pd.merge(nf,dates[['calendar_date','holiday_flg']],how = 'left', left_on='rd',right_on='calendar_date')
  nf = nf.rename(index = str, columns = {'holiday_flg':'reservation_holiday'})
  nf = nf.drop(['calendar_date'],axis=1)

  nf['vd'] = pd.to_datetime(nf['vd']).dt.dayofyear
  nf['rd'] = pd.to_datetime(nf['rd']).dt.dayofyear
  print(nf.head())

  return nf

def nf_plot(nf):
  deltatime = vdt - rdt
  days = deltatime.dt.days
  print(days.describe())

  f,axa = plt.subplots(1,1,figsize=(15,6))
  sns.distplot(days)
  plt.xlim(0,40)
  axa.set_title('Days between Reservation and Visit')
  plt.show()

  f,ax = plt.subplots(1,1, figsize=(15,6))
  vholidayhist= nf[nf['visit_holiday']==1].groupby(['vd'],as_index=False).count()
  sns.barplot(x = vholidayhist.vd,y=vholidayhist.visit_datetime)
  ax.set_title('Visits in Japanese Holidays')
  plt.show()

  f,ax = plt.subplots(1,1, figsize=(15,6))
  vholidayhist= nf[nf['visit_holiday']==0].groupby(['vd'],as_index=False).count()
  sns.barplot(x = vholidayhist.vd,y=vholidayhist.visit_datetime)
  ax.set_title('Visits in Other Days')
  plt.show()

def visitors_visulization(path='./data/air_visit_data.csv'):
  air_visit = pd.read_csv(path)
  air_visit['visit_date'] = pd.to_datetime(air_visit['visit_date'])

  # time series plot:
  p1 = air_visit.groupby('visit_date').sum()
  p1.plot()
  # plt.title('visited visitors')
  # plt.show()

  # frequency of visitors??

  # plot with day of week
  air_visit['dow'] = air_visit['visit_date'].dt.dayofweek
  p3 = air_visit.groupby('dow').median()
  p3.plot()
  # plt.title('median of day of the week')
  # plt.show()

  # plot with month
  p4 = air_visit['visitors'].groupby(air_visit['visit_date'].dt.month).median()
  p4.plot()
  # plt.title('median of month')
  # plt.show()

def air_analysis(air):
  pass

def holiday_analysis(path='./data/date_info.csv'):
  date_info = pd.read_csv(path, index_col=0)
  air_holiday_flg = []
  for i in range(len(air)):
    air_holiday_flg.append(date_info[date_info.index == air.loc[i, 'vivist_date']]['holiday_flg'].values[0])
  air['holiday_flg'] = air_holiday_flg
  # Should I put F, S, S as holiday?? seems need to do some visulization.

  # plot day of week and holiday analysis
  p0 = air_visit.loc[air_visit['holiday_flg']==0].groupby('dow').mean()
  p1 = air_visit.loc[air_visit['holiday_flg']==1].groupby('dow').mean()

def main():
  pass

if __name__ == '__main__':
  main()
