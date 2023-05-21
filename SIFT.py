import cv2
import os
from skimage import io
import time
import joblib
from skimage.filters import median
from skimage.exposure import equalize_hist
from skimage.feature import sift
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.model_selection import train_test_split


train_folder = "img/train"
test_folder = "img/test"
emotion_folders = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


images = []
for emotion_folder in emotion_folders:
    emotion_folder_path = os.path.join(train_folder, emotion_folder)
    for image_file in os.listdir(emotion_folder_path)[:10]:
        image_path = os.path.join(emotion_folder_path, image_file)
        label = emotion_folders.index(emotion_folder)
        images.append((image_path, label))

images_test = []
for emotion_folder in emotion_folders:
    emotion_folder_path = os.path.join(test_folder, emotion_folder)
    for image_file in os.listdir(emotion_folder_path)[:10]:
        image_path = os.path.join(emotion_folder_path, image_file)
        label = emotion_folders.index(emotion_folder)
        images_test.append((image_path, label))


#########################################################
################IMPLEMENTACIÓ SIFT#######################
#########################################################

final_train = []

emotions = []
i = 0
for path, emotion in images:

    img = io.imread(path)
    img = median(img)
    img = equalize_hist(img)

    # Extraemos el vector de características utilizando SIFT
    #sift = cv2.SIFT_create()
    #keypoints, descriptors = sift.detectAndCompute(img, None)
    keypoints, descriptors = sift(img)

    emotions.append(emotion)

    final_train.append(descriptors)
    if i == 0:
        print(final_train[0])
    i += 1


emotions_test = []
final_test = []

for path, emotion in images_test:

    img = io.imread(path)
    img = median(img)
    img = equalize_hist(img)

    # Extraemos el vector de características utilizando SIFT
    #sift = cv2.SIFT_create()
    #keypoints, descriptors = sift.detectAndCompute(img, None)
    keypoints, descriptors = sift(img)

    emotions_test.append(emotion)

    final_test.append(descriptors)

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

rfc = RandomForestClassifier()
#rfc = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=9,max_features=None, min_samples_leaf=1, min_samples_split=8, n_estimators=47)


rfc.fit(X, y)
accuracy = rfc.score(X_test, y_test)
print("Accuracy:", accuracy*100)