import tensorflow as tf
import numpy as np
from tensorflow import keras

# load data
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (testing_images, test_labels) = mnist.load_data()

# normalize data
training_images = training_images / 255.0
testing_images = testing_images / 255.0

# reshape training_images from (60000,28,28) to (60000,28,28,1)
training_images = training_images.reshape(len(training_images), 28, 28, 1)
testing_images = testing_images.reshape(len(testing_images), 28, 28, 1)


# define callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.998):
            print('\n Reached 90% accuracy so cancelling training!')
            self.model.stop_training = True


callbacks = myCallback()

# define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# fit
model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])
test_loss, test_acc = model.evaluate(testing_images, test_labels)
print("test_loss", test_loss, "test_acc", test_acc)
