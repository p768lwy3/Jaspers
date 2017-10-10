# Import:
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import random
import re
import tempfile
import tensorflow as tf

"""
  Taking the len of name.
  Since the upper class may have a longer name?
  Actually it is better to do a table for checking whether they are upper class..."""
def data_cleaning(ds):

  # 1. Removing Brackets
  ds['Name'] = ds['Name'].str.replace(r'[\(\[].*?[\)\]]', '')
  """
    OR Remaining the brackets only?
    re.sub('[(.*?{]', '', string)
  """
  # 2. Length of Name
  ds['Name'] = ds['Name'].str.split().str.len() - 1
  """Ticket No...?"""
  # 1. Check if there are chars in Ticket column
  ds['TicketWithChar'] = ds['Ticket'].str.contains(r'[A-Za-z]').astype('int')
  # 2. Length of Ticket No?
  ds['Ticket'] = ds['Ticket'].str.split()
  ds['Ticket'] = ds['Ticket'].map(lambda x: x[1] if len(x)>1 else x[0])
  ## Something call line...? You are mother Fucker?
  ds['Ticket'] = ds['Ticket'].replace('.', '').str.len()

  """Room No."""
  ds['Cabin'] = ds['Cabin'].str.split().str.len()

  """Replace Nan"""
  ds['Age'] = ds['Age'].fillna(value=round(ds['Age'].mean(),1))
  ds['Cabin'] = ds['Cabin'].fillna(value=0)
  #ds['Embarked'] = ds['Embarked'].fillna(value='Z')

  return ds

class NN_model(object):
  """ Deep and wide network """
  def __init__(self, CrossValid = True, RealTest = False):
    self.ds_train = pd.read_csv('train.csv')
    self.ds_train = data_cleaning(self.ds_train)
    if CrossValid == True:
      self.X_train = self.ds_train.sample(frac=0.7)
      self.X_test = self.ds_train.loc[~self.ds_train.index.isin(self.X_train.index)]
    else:
      self.X_train = self.ds_train
    if RealTest == True:
      self.ds_test = pd.read_csv('test.csv')
      self.ds_test = data_cleaning(self.ds_test)

  def input_fn(self, df, CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, LABEL_COLUMN):
    continuous_cols = {k: tf.constant(df[k].values)
      for k in CONTINUOUS_COLUMNS}
    categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values, dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
    feature_cols = continuous_cols.copy()
    feature_cols.update(categorical_cols)
    label = tf.convert_to_tensor(df[LABEL_COLUMN].values, dtype=tf.float32)
    return feature_cols, label

  def input_fn_predict(self, df, CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS):
    continuous_cols = {k: tf.constant(df[k].values)
      for k in CONTINUOUS_COLUMNS}
    categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values, dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
    feature_cols = continuous_cols.copy()
    feature_cols.update(categorical_cols)
    return feature_cols


  def neuralNetwork(self, CrossValid = True, RealTest = False):
    LABEL_COLUMN = 'Survived'
    CATEGORICAL_COLUMNS = ["Sex", "Embarked"]
    CONTINUOUS_COLUMNS = ["Pclass", "Age", "Fare", "Cabin"]
   #'Name', 'TicketWithChar', "Ticket", "SibSp", "Parch", 
    print('  Getting the Dataset Now......')
    """Categorical Base Columns"""
    categorical = {}
    for i in CATEGORICAL_COLUMNS:
      self.X_train[i] = self.X_train[i].apply(str)
      if CrossValid == True:
        self.X_test[i] = self.X_test[i].apply(str)
      if RealTest == True:
        self.ds_test[i] = self.ds_test[i].apply(str)
      keys = self.X_train[i].unique()
      # Keys of training set may not cover some of the test set... I may need to do some convert of data.
      categorical[i] = tf.contrib.layers.sparse_column_with_keys(column_name=i, keys=keys)

    """Continuous Base Columns"""
    continuous = {}
    for i in CONTINUOUS_COLUMNS:
      self.X_train[i] = pd.to_numeric(self.ds_train[i], errors='coerce')
      if CrossValid == True:
        self.X_test[i] = pd.to_numeric(self.X_test[i], errors='coerce')
      if RealTest == True:
        self.ds_test[i] = pd.to_numeric(self.ds_test[i], errors='coerce')
      continuous[i] = tf.contrib.layers.real_valued_column(i)
    continuous['Age_buckets'] = tf.contrib.layers.bucketized_column(continuous['Age'], 
                                  boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    print('  Building the Network Now......')
    wide_columns = []; deep_columns = []
    for i in CATEGORICAL_COLUMNS:
      wide_columns.append(categorical[i])
      deep_columns.append(tf.contrib.layers.embedding_column(categorical[i], 
                                                             dimension=len(categorical[i])))
    for i in CONTINUOUS_COLUMNS:
      deep_columns.append(continuous[i])

    """ Boosting..."""
    print('  Doing Boosting Now......')
    wide_columns.append(tf.contrib.layers.crossed_column([categorical['Sex'],
                          categorical['Embarked']], hash_bucket_size=int(1e4)))
    wide_columns.append(tf.contrib.layers.crossed_column([categorical['Sex'],
                          continuous['Age_buckets']],  hash_bucket_size=int(1e4)))
    #for i in range(3):
      #randlist = random.sample(CATEGORICAL_COLUMNS, 3)
      #print('    CATEGORICAL COLUMNS: %s, %s, %s are choosen...' % (randlist[0], randlist[1], randlist[2]))
      #c1, c2, c3 = categorical[randlist[0]], categorical[randlist[1]], categorical[randlist[2]]
      #wide_columns.append(tf.contrib.layers.crossed_column([c1, c2, c3], hash_bucket_size=int(1e4)))
    

    model_dir = tempfile.mkdtemp()
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
      model_dir=model_dir,
      linear_feature_columns=wide_columns,
      dnn_feature_columns=deep_columns,
      dnn_hidden_units=[128, 64])

    def input_fn_train():
      return self.input_fn(self.X_train, CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, LABEL_COLUMN)
    def input_fn_crossvalid():
      return self.input_fn(self.X_test, CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, LABEL_COLUMN)
    def input_fn_test():
      return self.input_fn_predict(self.ds_test, CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS)

    m.fit(input_fn=input_fn_train, steps=1000)
    if CrossValid == True:
      crossvalid_results = m.evaluate(input_fn=input_fn_crossvalid, steps=1)
      for key in sorted(crossvalid_results):
        print(  "%s: %s" % (key, crossvalid_results[key]))
    if RealTest == True:
      test_result = m.predict(input_fn=input_fn_test, as_iterable=False)
      print(test_result)
      """ Save result to .csv """
      ds_result = pd.read_csv('gender_submission.csv', index_col=0)
      ds_result['Survived'] = test_result
      print(ds_result.head(10))
      ds_result.to_csv('gender_submission.csv')

def model_(split = True):
  """ Read dataset  """
  ds_train = pd.read_csv('train.csv')
  for i in ['Sex', 'Embarked']:
    enum = {list(set(ds_train[i]))[j]:j for j in range(len(set(ds_train[i])))}
    ds_train[i] = ds_train[i].map(enum)
  print(ds_train)
  print(ds_train.dtypes)
  ds_train = ds_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
  ds_train = ds_train.dropna(axis=0, how='any')

  #X_train = ds_train.drop(['Survived'], axis=1)
  X_train = ds_train[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
  y_train = ds_train['Survived']
  if split == True:
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4)
  elif split == False:
    ds_test = pd.read_csv('test.csv')
    X_test = ds_test
  
  model = SVC(C=1.0, kernel='poly', degree=3).fit(X_train, y_train)
  print(model.score(X_test, y_test))

if __name__ == '__main__':
  #model_()
  nnmodel = NN_model(CrossValid=False, RealTest=True)
  #print(nnmodel.X_train.head(10))
  #print(pd.isnull(nnmodel.X_train).sum())
  nnmodel.neuralNetwork(CrossValid=False, RealTest=True)
