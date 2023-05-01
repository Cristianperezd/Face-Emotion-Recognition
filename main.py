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


train_folder = "img/train"
emotion_folders = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


images = []
for emotion_folder in emotion_folders:
    emotion_folder_path = os.path.join(train_folder, emotion_folder)
    for image_file in os.listdir(emotion_folder_path):
        image_path = os.path.join(emotion_folder_path, image_file)
        label = emotion_folders.index(emotion_folder)
        images.append((image_path, label))


# Definimos los parámetros de LBP
radius = 1
n_points = 8 * radius
method = 'uniform'



lbp_histograms = []
emotions = []
i = 0
for path,emotion in images:

    img = io.imread(path)

    lbp = local_binary_pattern(img, n_points, radius, method)

    # Extraemos un histograma de los patrones binarios locales
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    # Normalizamos el histograma para que sume 1
    hist /= np.sum(hist)
    lbp_histograms.append(hist)
    emotions.append(emotion)
    if i == 0:
        print(lbp_histograms[0][1])
    i = 2
print('Final LBP')

# Dividimos los datos en conjunto de entrenamiento y de prueba
"""
X = np.array(lbp_histograms)
y = np.array(emotions)"""

# Obtener la longitud máxima de los histogramas
max_len = max(len(hist) for hist in lbp_histograms)

# Rellenar los histogramas con ceros al final para que tengan la misma longitud
padded_histograms = []
for hist in lbp_histograms:
    padded_hist = np.pad(hist, (0, max_len - len(hist)), mode='constant')
    padded_histograms.append(padded_hist)

# Convertir los datos de entrada a un array de NumPy homogéneo
X= np.array(padded_histograms)

# Convertir las emociones a un array de NumPy
y = np.array(emotions)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
# Creamos un clasificador KNN con 5 vecinos
knn = KNeighborsClassifier(n_neighbors=10)

# Entrenamos el clasificador con los datos de entrenamiento
knn.fit(X_train, y_train)

# Evaluamos el clasificador con los datos de prueba
accuracy = knn.score(X_test, y_test)
print("Accuracy:", accuracy)



# Guardar el modelo en un archivo llamado "modelo_knn.joblib"
joblib.dump(knn, "modelo_knn.joblib")"""

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)

