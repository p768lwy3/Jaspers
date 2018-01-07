import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

train = pd.read_json('./data/train.json')
test = pd.read_json('./data/test.json')
X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train['band_1']])
X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train['band_2']])
X_train = np.concatenate([X_band_1[:,:,:,np.newaxis], X_band_2[:,:,:,np.newaxis], ((X_band_1+X_band_2)/2)[:,:,:,np.newaxis]], axis=-1)

pic = X_band_1[12,:,:]
z = pic[::1, ::1]
x, y = np.mgrid[:z.shape[0], :z.shape[1]]
title = 'iceberg'

# 2d color map
fig = plt.figure()
plt.imshow(z)
plt.title(title)
plt.show()

# 3d surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title(title)
plt.show()
