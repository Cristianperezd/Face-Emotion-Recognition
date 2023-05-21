from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from face_rec import FaceRecognition
import matplotlib.pyplot as plt
import os 
from skimage import io

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta al archivo h5 del modelo guardado
model_path = 'Model2.h5'

# Cargar el modelo
model = load_model(model_path)

"""
train_folder = "img/train"
test_folder = "img/test"
emotion_folders = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Ruta al archivo h5 del modelo guardado
model_path = 'Model2.h5'




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

X_test = []
y_test = []

for path,emotion in images_test:

    img = io.imread(path)
    img = image.load_img(path, target_size=(48, 48), grayscale=True)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    
    X_test.append(img_array)
    y_test.append(emotion)
   

# Ruta al archivo h5 del modelo guardado
model_path = 'Model2.h5'

# Cargar el modelo
model = load_model(model_path)

"""



"""

y_pred = []
for img in X_test:
    y_pred_prob = model.predict(img)
    y_pred.append(np.argmax(y_pred_prob, axis=1)) 
  

cm = confusion_matrix(y_test,y_pred)

fig, ax = plt.subplots()

class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Crea el mapa de calor utilizando la matriz de confusión
heatmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)

# Establece etiquetas para los ejes x e y
ax.set_xlabel('Predicted Emotions')
ax.set_ylabel('True Emotions')

# Establece el título del gráfico
ax.set_title('Confusion Matrix')

# Muestra el mapa de calor
plt.show()

"""


face_rec = FaceRecognition()

img_path = 'pique_manita.jpg'

for face in face_rec.detect_faces(img_path):

    # Cargar la imagen de prueba
    
    # Preprocesar la imagen para que coincida con el formato esperado por el modelo
    img_array = np.array(face)
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar la predicción utilizando el modelo cargado
    predictions = model.predict(img_array)

    probabilities = predictions[0]
 
    # Obtener la etiqueta de la emoción predicha
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    predicted_label = emotions[np.argmax(predictions)]
    for emotion, probability in zip(emotions, probabilities):
        print(f"{emotion}: {probability * 100:.2f}%")
    # Imprimir el resultado
    print("Emotion predicted:", predicted_label)
    plt.imshow(face, cmap='gray')
    plt.title("Emotion predicted: " + predicted_label)
    plt.axis("off")
    plt.show()





"""# Cargar la imagen de prueba
img_path = 'img/test/angry/PrivateTest_88305.jpg'

img = image.load_img(img_path, target_size=(48, 48), grayscale=True)

# Preprocesar la imagen para que coincida con el formato esperado por el modelo
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Realizar la predicción utilizando el modelo cargado
predictions = model.predict(img_array)
print(predictions)

# Obtener la etiqueta de la emoción predicha
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
predicted_label = emotions[np.argmax(predictions)]

# Imprimir el resultado
print("Emotion predicted:", predicted_label)

# Imprimir el resultado
print("Emotion predicted:", predicted_label)
plt.imshow(img, cmap='gray')
plt.title("Emotion predicted: " + predicted_label)
plt.axis("off")
plt.show()"""

