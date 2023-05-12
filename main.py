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
import sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV

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


# Definimos los parámetros de LBP
radius = 1
n_points = 8 * radius
method = 'uniform' 



lbp_histograms = []
emotions = []
i = 0
for path,emotion in images:

    img = io.imread(path)
    img = median(img)
    img = equalize_hist(img)

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
lbp_histograms_test = []
emotions_test = []
for path,emotion in images_test:

    img = io.imread(path)
    img = median(img)
    img = equalize_hist(img)



    lbp = local_binary_pattern(img, n_points, radius, method)

    # Extraemos un histograma de los patrones binarios locales
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    # Normalizamos el histograma para que sume 1
    hist /= np.sum(hist)
    lbp_histograms_test.append(hist)
    emotions_test.append(emotion)
    if i == 0:
        print(lbp_histograms_test[0][1])
    i = 2


# Dividimos los datos en conjunto de entrenamiento y de prueba
"""
X = np.array(lbp_histograms)
y = np.array(emotions)"""

#Train

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

#Test

max_len_test = max(len(hist) for hist in lbp_histograms_test)

# Rellenar los histogramas con ceros al final para que tengan la misma longitud
padded_histograms_test = []
for hist in lbp_histograms_test:
    padded_hist = np.pad(hist, (0, max_len_test - len(hist)), mode='constant')
    padded_histograms_test.append(padded_hist)

# Convertir los datos de entrada a un array de NumPy homogéneo
X_test= np.array(padded_histograms_test)

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

# Define los hiperparámetros que deseas ajustar y sus rangos
param_dist = {
    'n_estimators': randint(10, 100),
    'max_depth': randint(2, 10),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# Define el modelo de Random Forest
"""rfc = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=9,max_features=None, min_samples_leaf=1, min_samples_split=8, n_estimators=47)

# Realiza la búsqueda aleatoria de hiperparámetros
random_search = RandomizedSearchCV(
    rfc,
    param_distributions=param_dist,
    n_iter=50, # Número de iteraciones aleatorias
    cv=5, # Número de folds para cross-validation
    scoring='accuracy', # Métrica a optimizar
    n_jobs=-1, # Número de núcleos a utilizar (-1 utiliza todos los disponibles)
    verbose=1, # Nivel de detalle en la salida
    random_state=42 # Semilla aleatoria para reproducibilidad
)

# Entrena el modelo con la búsqueda de hiperparámetros
random_search.fit(X, y)
 ámetros encontrados y su puntuación
print("Mejores hiperparámetros encontrados:")
print(random_search.best_params_)
print("Puntuación de validación cruzada del mejor modelo:")
print(random_search.best_score_)

"""
rfc = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators' : [50,100,200],
    'max_depth' : [None, 5,10,20],
    'min_samples_split' : [2,5,10],
    'min_samples_leaf' : [1,2,4],
    'max_features' : ['sqrt','log2']
}
grid_search = GridSearchCV(estimator=rfc,param_grid=param_grid, cv=5,n_jobs=1,verbose=2)

grid_search.fit(X,y)

print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)
print("Puntuación de validación cruzada del mejor modelo:")
print(grid_search.best_score_)


"""rfc = RandomForestClassifier()
#rfc = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=9,max_features=None, min_samples_leaf=1, min_samples_split=8, n_estimators=47)


rfc.fit(X, y)
accuracy = rfc.score(X_test, y_test)
print("Accuracy:", accuracy*100)"""

"""
# Guardar el modelo en un archivo llamado "modelo_knn.joblib"
joblib.dump(knn, "modelo_knn.joblib")"""

"""clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X, X_test, y, y_test)
print(models)
"""
