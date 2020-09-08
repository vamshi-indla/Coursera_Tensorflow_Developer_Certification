# source
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=RXZT2UsyIVe_
# https://hsmuxpcjyabtszqkvkhxkm.coursera-apps.org/notebooks/week1/Exercise_1_Cats_vs_Dogs_Question-FINAL.ipynb
# split training and validation

import zipfile, os
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

zip_ref = zipfile.ZipFile('/tmp/cats_and_dogs_filtered.zip', 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

base_dir = ('/tmp/cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# directory with cat and dog training
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

print('total of training cat images:', len(os.listdir(train_cats_dir)))
print('total of training dog images:', len(os.listdir(train_dogs_dir)))
print('total of validation cat images:', len(os.listdir(validation_cats_dir)))
print('total of validation dog images:', len(os.listdir(validation_cats_dir)))


# define callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.70):
            print('\n Reached 70% accuracy so cancelling training!')
            self.model.stop_training = True


callbacks = myCallback()

# tensorflow
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

])

# compile
model.compile(loss='binary_crossentropy', optimizer=RMSprop(0.001), metrics=['accuracy'])

#
BATCH_SIZE = 128
EPOCHS = 20
TRAINING_IMAGES_COUNT = len(os.listdir(train_cats_dir)) + len(os.listdir(train_dogs_dir))
VALIDATION_IMAGES_COUNT = len(os.listdir(validation_cats_dir)) + len(os.listdir(validation_dogs_dir))

# train_datagen = ImageDataGenerator(rescale=1/255.0)
train_datagen = ImageDataGenerator(rescale=1 / 255.0,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest'
                                   )

validation_datagen = ImageDataGenerator(rescale=1 / 255.0)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(150, 150),
                                                              batch_size=BATCH_SIZE,
                                                              class_mode='binary')

# fit the model
history = model.fit(train_generator,
                    steps_per_epoch=int(TRAINING_IMAGES_COUNT / BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=validation_generator,
                    validation_steps=int(VALIDATION_IMAGES_COUNT / BATCH_SIZE)
                    )

model.summary()

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
