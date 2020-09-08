# source
# https://btekvptxgpyekhuwwqhwja.coursera-apps.org/notebooks/week4/Exercise4-Question.ipynb

# load packages
import tensorflow as tf
import zipfile, os
from os import getcwd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

# load dataset
path = f"{getcwd()}/tmp/happy-or-sad.zip"
print(path)

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall('/tmp/h-or-s')
zip_ref.close()


# define function to preprocess, define, compile and fit model

def train_happy_sad_model():
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > 0.999):
                print('\n Reached 99.9% accuracy so cancelling training!')
                self.model.stop_training = True

    DESIRED_ACCURACY = 0.999
    batch_size = 8
    callbacks = myCallback()

    train_datagen = ImageDataGenerator(1 / 255.0)
    train_generator = train_datagen.flow_from_directory('/tmp/h-or-s', target_size=(150, 150), batch_size=batch_size,
                                                        class_mode='binary')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(0.001), metrics=['accuracy'])

    history = model.fit(train_generator, epochs=20, callbacks=[callbacks], verbose=1)
    return history.history['accuracy'][-1]


# invoke function
train_happy_sad_model()
