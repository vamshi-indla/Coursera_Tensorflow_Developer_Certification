import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# source
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%202.ipynb

"""
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json \
    -O /tmp/sarcasm.json

Bidirectional LSTM
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202.ipynb#scrollTo=g9DC6dmLF8DC

CONV1D
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202c.ipynb#scrollTo=g9DC6dmLF8DC
"""

'''
binary classification

'''


# Variables
EPOCHS = 20
VOCAB_SIZE = 15000
OOV_TOKEN = "oov"
MAX_LENGTH = 40
EMBEDDING_DIM = 32
TRAINING_SIZE = 20000

with open("/tmp/sarcasm.json") as f:
    datastore = json.load(f)

sentences = []
urls = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

train_sentences = sentences[0:TRAINING_SIZE]
train_labels = np.array(labels[0:TRAINING_SIZE])
test_sentences = sentences[TRAINING_SIZE:]
test_labels = np.array(labels[TRAINING_SIZE:])

# preprocessing


tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(train_sentences)
train_seq = tokenizer.texts_to_sequences(train_sentences)
test_seq = tokenizer.texts_to_sequences(test_sentences)

train_padded = pad_sequences(train_seq, maxlen=MAX_LENGTH, truncating='post')
test_padded = pad_sequences(test_seq, maxlen=MAX_LENGTH, truncating='post')

print(train_padded.shape, test_padded.shape)

# define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    # tf.keras.layers.Flatten(),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(train_padded, train_labels,
                    epochs=EPOCHS,
                    validation_data=(test_padded, test_labels))

# plotting


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
EPOCHS_RANGE = range(len(acc))

plt.subplot(1, 2, 1)
plt.plot(EPOCHS_RANGE, acc, label='Training Accuracy')
plt.plot(EPOCHS_RANGE, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(EPOCHS_RANGE, loss, label='Training Loss')
plt.plot(EPOCHS_RANGE, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.show()
print(history.history['accuracy'])
