import tensorflow as tf
import numpy as np


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.99):
            print('\n Reached 99% accuracy so cancelling training!')
            self.model.stop_training = True


def train_mnist():
    callbacks = myCallback()

    # load data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalized
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # define model
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                        tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    # model compile

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fit
    history = model.fit(x_train, y_train, epochs=5, callbacks=[callbacks])
    return history.epoch, history.history['accuracy'][-1]


train_mnist()
