## Warm up with dataframe and datareader
import pandas_datareader.data as dta
import string as str
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def pause():
  programPause = input('Program paused. Press <Enter> to continue.')

def main():
  code = ['DEXUSEU', 'DEXCHUS', 'DEXCAUS', 'DEXMXUS', 'DEXJPUS']

  ## crawl data and build up dataframe and its matrix
  df = dta.DataReader('GDP', 'fred')
  for x in code:
    temp = dta.DataReader(x, 'fred')
    df = pd.concat([df, temp], axis=1, join='inner')
  df = df[np.isfinite(df['DEXJPUS'])]
  df = df.values

  ## Train_data and training model
  X, y = df[0:14,1:5], df[0:14,0]
  reg = MLPRegressor(hidden_layer_sizes=(50,), solver='lbfgs').fit(X,y)
  for i in range(14, 17):
    predict = float(reg.predict(df[i,1:5].reshape(1,-1)))
    print('predict: ', '{0:.4f}'.format(predict), 'real: ', df[i,0], 'Error: ', '{0:.4f}'.format(predict-df[i,0]), 
          'Error%: ', '{0:.2f}'.format((predict-df[i,0])/df[i,0]*100), '%')
  pause()

if __name__ == "__main__":
  main()
