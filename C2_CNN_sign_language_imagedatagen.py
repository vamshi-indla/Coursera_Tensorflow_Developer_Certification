# source
# #https://hsmuxpcjyabtszqkvkhxkm.coursera-apps.org/notebooks/week4/Exercise_4_Multi_class_classifier_Question-FINAL.ipynb
'''
Mutliclass
CNN
imagedatagenerator
preprocessing csv
NO flow_from_directory
'''

from os import getcwd
import os, numpy as np

base_dir = '/Users/vamshi294/Tensorflow_certification/'
train_path = os.path.join(base_dir, 'sign_mnist_train.csv')
validation_path = os.path.join(base_dir, 'sign_mnist_test.csv')


def get_data(filename):
    labels = []
    images = []
    with open(filename) as training_file:
        first = 1
        for line in training_file.readlines():
            if first == 1:
                first = 0
            else:
                lbl = line[0]
                img = line.split(',')[1:]
                labels.append(lbl)
                images.append(np.array_split(img, 28))
    labels = np.array(labels).astype('int32')
    images = np.array(images).astype('float32')
    return images, labels


training_images, training_labels = get_data(train_path)
validation_images, validation_labels = get_data(validation_path)

print(training_labels.shape)
print(training_images.shape)
print(validation_labels.shape)
print(validation_images.shape)

# data preprocessing
training_images = training_images.reshape(len(training_images), 28, 28, 1)
validation_images = validation_images.reshape(len(validation_images), 28, 28, 1)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    shear_range=0.2,
    width_shift_range=0.2,
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(1 / 255.0)

from tensorflow.keras.utils import to_categorical

training_labels = to_categorical(training_labels)
validation_labels = to_categorical(validation_labels)

# define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(9, activation=tf.nn.softmax)
])
# model compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model fit
EPOCHS = 5
BATCH_SIZE = 32
history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=BATCH_SIZE),
                              epochs=EPOCHS,
                              validation_data=(validation_images, validation_labels),
                              verbose=1)
# plot
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training accuracy')
plt.plot(epochs_range, val_acc, label='Validation accuracy')
plt.legend()
plt.title('Training and validation accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='Validation loss')
plt.legend()
plt.title('Training and validation accuracy')

plt.show()
