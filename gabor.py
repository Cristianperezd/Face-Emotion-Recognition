import cv2
import os
from skimage import io
import time
import joblib
from skimage.filters import median
from skimage.exposure import equalize_hist

from skimage.filters import gabor_kernel
from scipy import ndimage as ndi



from lazypredict.Supervised import LazyClassifier


from skimage.feature import local_binary_pattern
import numpy as np


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



train_folder = "img/train"
test_folder = "img/test"
emotion_folders = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


images = []
for emotion_folder in emotion_folders:
    emotion_folder_path = os.path.join(train_folder, emotion_folder)
    for image_file in os.listdir(emotion_folder_path):
        image_path = os.path.join(emotion_folder_path, image_file)
        label = emotion_folders.index(emotion_folder)
        images.append((image_path, label))

images_test = []
for emotion_folder in emotion_folders:
    emotion_folder_path = os.path.join(test_folder, emotion_folder)
    for image_file in os.listdir(emotion_folder_path):
        image_path = os.path.join(emotion_folder_path, image_file)
        label = emotion_folders.index(emotion_folder)
        images_test.append((image_path, label))

frequencies = [0.1, 0.2, 0.4]
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Creamos los kernels de Gabor
kernels = []
for frequency in frequencies:
    for theta in orientations:
        kernel = np.real(gabor_kernel(frequency, theta=theta))
        kernels.append(kernel)

final_train = []

emotions = []
for path,emotion in images:

    img = io.imread(path)
    img = median(img)
    img = equalize_hist(img)

    results = []
    for kernel in kernels:
        filtered = ndi.convolve(img, kernel, mode='wrap')
        results.append(filtered)

        # Concatenamos los resultados en una sola matriz de características
    feature_matrix = np.concatenate([r.flatten() for r in results])

    # Normalizamos los valores de las características entre 0 y 1
    feature_matrix = (feature_matrix - feature_matrix.min()) / (feature_matrix.max() - feature_matrix.min())

    emotions.append(emotion)

    final_train.append(feature_matrix)
    


emotions_test = []
final_test = []

for path,emotion in images_test:

    img = io.imread(path)
    img = median(img)
    img = equalize_hist(img)

    results = []
    for kernel in kernels:
        filtered = ndi.convolve(img, kernel, mode='wrap')
        results.append(filtered)

        # Concatenamos los resultados en una sola matriz de características
    feature_matrix = np.concatenate([r.flatten() for r in results])

    # Normalizamos los valores de las características entre 0 y 1
    feature_matrix = (feature_matrix - feature_matrix.min()) / (feature_matrix.max() - feature_matrix.min())

    emotions_test.append(emotion)

    final_test.append(feature_matrix)



########################################################################






##########################################################################




# Convertir los datos de entrada a un array de NumPy homogéneo
X= np.array(final_train)

# Convertir las emociones a un array de NumPy
y = np.array(emotions)

#Test



# Convertir los datos de entrada a un array de NumPy homogéneo
X_test= np.array(final_test)

# Convertir las emociones a un array de NumPy
y_test = np.array(emotions_test)


"""
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos un clasificador KNN con 5 vecinos
knn = KNeighborsClassifier(n_neighbors=5)

# Entrenamos el clasificador con los datos de entrenamiento
knn.fit(X, y)

# Evaluamos el clasificador con los datos de prueba
accuracy = knn.score(X_test, y_test)
print("Accuracy:", accuracy*100)
"""

"""
# Guardar el modelo en un archivo llamado "modelo_knn.joblib"
joblib.dump(knn, "modelo_knn.joblib")"""

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X, X_test, y, y_test)
print(models)
