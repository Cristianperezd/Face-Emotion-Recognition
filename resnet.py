import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
"""for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))"""


import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow import keras

from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model


train_dir = "img/train"
test_dir = "img/test"

train_datagenerator = ImageDataGenerator( rescale = 1./255,
                                         width_shift_range = 0.1,
                                         height_shift_range = 0.1,
                                         horizontal_flip = True,)
validation_datagenerator = ImageDataGenerator( rescale = 1./255,)

train_generator = train_datagenerator.flow_from_directory(directory = train_dir,
                                                         target_size = (48,48),
                                                         batch_size = 64,
                                                         
                                                         class_mode = "categorical")

validation_generator = validation_datagenerator.flow_from_directory(directory = test_dir,
                                                        target_size = (48,48),
                                                         batch_size = 64,
                                                         
                                                         class_mode = "categorical")


epochs = 20

resnet_model = keras.models.Sequential()
pretrained_model = tf.keras.applications.ResNet50(include_top = False, 
                                                        input_shape = (48,48,3),
                                                       pooling = 'avg',
                                                       classes = 7,
                                                       weights='imagenet')
for each_layer in pretrained_model.layers:
    each_layer.trainable = False
resnet_model.add(pretrained_model)

resnet_model.add(Flatten())
resnet_model.add(Dense(512,activation='relu'))
resnet_model.add(Dense(7,activation='softmax'))

resnet_model.compile(
    optimizer= Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
resnet_model.summary()

checkpoint = ModelCheckpoint("resnet_trained.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = resnet_model.fit(x = train_generator, epochs = epochs, validation_data = validation_generator, callbacks=[checkpoint])
