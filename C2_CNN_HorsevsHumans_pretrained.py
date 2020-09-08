# source
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb#scrollTo=ClebU9NJg99G

# wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip  -O /tmp/horse-or-human.zip
# wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip -O /tmp/validation-horse-or-human.zip

# pretrained inception V3
# CNN + binary classification + ImageGenerator + Dropout

import os, zipfile

local_zip = '/tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/horse-or-human')
local_zip = '/tmp/validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation-horse-or-human')
zip_ref.close()

train_horse_dir = os.path.join('/tmp/horse-or-human/horses')
train_human_dir = os.path.join('/tmp/horse-or-human/humans')
validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')
validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')

print('total training horse images: ', len(os.listdir(train_horse_dir)))
print('total training human images: ', len(os.listdir(train_human_dir)))
print('total validation horse images: ', len(os.listdir(validation_horse_dir)))
print('total validation human images: ', len(os.listdir(validation_human_dir)))

# import packages
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop

# define model
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
#     tf.keras.layers.MaxPool2D(2,2),
#     tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512,activation='relu'),
#     tf.keras.layers.Dense(1,activation='sigmoid')
# ])

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model

pre_trained_model = InceptionV3(input_shape=(300, 300, 3),
                                include_top=False,
                                weights='imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(128, activation=tf.nn.relu)(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
# pre_trained_model.summary()
# model.summary()

# compile model
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

# fit model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1 / 255.)
validation_datagen = ImageDataGenerator(rescale=1 / 255.)

BATCH_SIZE = 20
EPOCHS = 5
TRAIN_SIZE = len(os.listdir(train_horse_dir)) + len(os.listdir(train_human_dir))
VALIDATION_SIZE = len(os.listdir(validation_horse_dir)) + len(os.listdir(validation_human_dir))

train_generator = train_datagen.flow_from_directory(
    '/tmp/horse-or-human/',
    target_size=(300, 300),
    batch_size=BATCH_SIZE,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    '/tmp/validation-horse-or-human/',
    target_size=(300, 300),
    batch_size=BATCH_SIZE,
    class_mode='binary')

# fit model
history = model.fit_generator(train_generator,
                              steps_per_epoch=int(TRAIN_SIZE / BATCH_SIZE),
                              epochs=EPOCHS,
                              verbose=1,
                              validation_data=validation_generator,
                              validation_steps=int(VALIDATION_SIZE / BATCH_SIZE))

# plotting
# plot accuracies and loss
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
