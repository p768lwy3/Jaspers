#import keras
#from keras.models import Sequential
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def gbr_fit(X_train, y_train, X_test, y_test):
  # Fit Regression Model:
  params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split':2,
    'learning_rate': 0.01, 'loss': 'ls'}
  clf = GradientBoostingRegressor(**params)

  clf.fit(X_train, y_train)
  print('MSE: %.4f' % (mean_squared_error(y_test, clf.predict(X_test))))

  # Plot training deviance
  test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
  for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)
  plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.title('Deviance')
  plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
    label='Training Set Deviance')
  plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
    label='Test Set Deviance')
  plt.legend(loc='upper right')
  plt.xlabel('Boosting Iterations')
  plt.ylabel('Deviance')

  # Plot feature importance
  feature_importance = clf.feature_importances_
  feature_importance = 100.0 * (feature_importance / feature_importance.max())
  sorted_idx = np.argsort(feature_importance)
  pos = np.arange(sorted_idx.shape[0]) + .5
  plt.subplot(1, 2, 2)
  plt.barh(pos, feature_importance[sorted_idx], align='center')
  plt.yticks(pos, X_train.columns[sorted_idx].values)
  plt.xlabel('Relative Importance')
  plt.title('Variable Importance')
  plt.show()
  
  return clf

def main():
  # check data types

  # data['dow'] = pd.to_datetime(data['visit_date']).dt.dayofweek

  # data = data.fillna(0)
  # dummy = pd.get_dummies(data['dummy'])
  # data = pd.concat([data, dummy], axis=1)
  # data = data.drop('one_dummy_columns', axis=1)

  # data = data.rename(columns={0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat'})

  y = data['visitors']
  X = data.drop('visitors', axis=1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

if __name__ == '__main__':
  main()

