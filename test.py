import cv2
import os
from skimage import io
import time
import joblib



from skimage.feature import local_binary_pattern
import numpy as np


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


img = io.imread('img/test/angry/PrivateTest_26530508.jpg', as_gray=True)

radius = 1
n_points = 8 * radius
method = 'uniform'
# Calcular el histograma LBP de la imagen
lbp = local_binary_pattern(img, n_points, radius, method)
n_bins = int(lbp.max() + 1)
hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

# Normalizar el histograma


hist /= np.sum(hist)


knn = joblib.load("modelo_knn.joblib")


# Hacer la predicción con el modelo entrenado
emotion_label = knn.predict_proba([hist])[0]

print("La emoción detectada en la imagen es:", emotion_label)
io.imshow(img)