import cv2
import os
from skimage import io
import time
import joblib

from lazypredict.Supervised import LazyClassifier


from skimage.feature import local_binary_pattern
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skimage.filters import median
from skimage.exposure import equalize_hist

import matplotlib.pyplot as plt



img = io.imread('img/test/angry/PublicTest_16131911.jpg', as_gray=True)


filter = median(img)

fig, axs = plt.subplots(3, 2, figsize=(10, 5))

axs[0,0].imshow(img, cmap='gray')
axs[0,0].set_title('Imagen ')
#axs[0,0].axis('off')

axs[0,1].imshow(filter, cmap='gray')
axs[0,1].set_title('Imagen median filter')
#axs[0,1].axis('off')

#plt.show()

filter_eq = equalize_hist(img)
filter_eqmedian = equalize_hist(filter)

#fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[1,0].imshow(filter_eq, cmap='gray')
axs[1,0].set_title('Imagen eq hist ')
#axs[1,0].axis('off')

axs[1,1].imshow(filter_eqmedian, cmap='gray')
axs[1,1].set_title('Imagen median + eq hist')
#axs[1,1].axis('off')

filter_ = median(filter_eq)

axs[2,0].imshow(filter_eq, cmap='gray')
axs[2,0].set_title('Imagen eq hist + median ')
plt.show()

