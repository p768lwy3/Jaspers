
"""
Ref:
  Data from: https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data
  Image Classifier: https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/
  Inception-V3: https://github.com/fchollet/keras/issues/6875
  Resnet: https://github.com/raghakot/keras-resnet
"""
## Import:
import cv2, json
import numpy as np
import pandas as pd


def save_pic(path='./data/train.json'):
  df = pd.read_json(path)
  for i, row in df.iterrows():
    img1 = np.array(row['band_1']).reshape(75, 75)
    img2 = np.array(row['band_2']).reshape(75, 75)
    img3 = img1 + img2
    img3 -= img3.min()
    img3 /= img3.max()
    img3 *= 255
    img3 = img3.astype(np.uint8)
    if row['is_iceberg'] == 0:
      cv2.imwrite('./train/0/f{}.png'.format(i), img3)
    elif row['is_iceberg'] == 1:
      cv2.imwrite('./train/1/f{}.png'.format(i), img3)
  
def main():
  save_pic()
  
if __name__ == '__main__':
  main()
