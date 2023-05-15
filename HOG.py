import cv2
import os
from skimage import io
import time
#import joblib
from skimage.filters import median
from skimage.exposure import equalize_hist

from skimage.feature import hog
import numpy as np

from lazypredict.Supervised import LazyClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier




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


final_train = []

emotions = []
i = 0
for path,emotion in images:

    img = io.imread(path)
    img = median(img)
    img = equalize_hist(img)

    # Extraemos el vector de características utilizando HOG
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True)

    emotions.append(emotion)

    final_train.append(fd)
    if i == 0:
    
        print(final_train[0])
    i = i + 1
    


emotions_test = []
final_test = []

for path,emotion in images_test:

    img = io.imread(path)
    img = median(img)
    img = equalize_hist(img)

    # Extraemos el vector de características utilizando HOG
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True)

    emotions_test.append(emotion)

    final_test.append(fd)



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
"""
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X, X_test, y, y_test)
print(models)
"""

rfc = RandomForestClassifier()
#rfc = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=9,max_features=None, min_samples_leaf=1, min_samples_split=8, n_estimators=47)


rfc.fit(X, y)
accuracy = rfc.score(X_test, y_test)
print("Accuracy:", accuracy*100)