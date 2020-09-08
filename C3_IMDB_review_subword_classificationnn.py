# source
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%203.ipynb

# conv1d (GRU)
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201c.ipynb

# GRU
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202d.ipynb#scrollTo=nHGYuU4jPYaj
'''
using subtoken
binary classification
LSTM Bidirectional
'''
# data prep
import tensorflow_datasets as tfds

imdb, info = tfds.load('imdb_reviews/subwords8k', as_supervised=True, with_info=True)

train, test = imdb['train'], imdb['test']
'''
train_sentences = []
test_sentences = []
train_labels = []
test_labels = []

for s, l in train:
    train_sentences.append(s.numpy().decode('utf8'))
    train_labels.append(s.numpy().decode('utf8'))

for s, l in test:
    test_sentences.append(s.numpy().decode('utf8'))
    test_labels.append(s.numpy().decode('utf8'))
'''

# preprocessing text
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

VOCAB_SIZE = 10000
OOV_TOKEN = "oov"
MAX_LENGTH = 40
EMBEDDING_DIM = 16
EPOCHS = 20
BUFFERSIZE = 10000
BATCH_SIZE = 64

tokenizer = info.features['text'].encoder
train_dataset = train.shuffle(BUFFERSIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train))
test_dataset = test.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test))

'''
tokenizer = Tokenizer(num_words=VOCAB_SIZE,oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(train_sentences)
train_seq = tokenizer.texts_to_sequences(train_sentences)
test_seq = tokenizer.texts_to_sequences(test_sentences)

train_padded = pad_sequences(train_seq,truncating='post',maxlen=MAX_LENGTH)
test_padded = pad_sequences(test_seq,truncating='post',maxlen=MAX_LENGTH)

train_padded = np.array(train_padded)
test_padded = np.array(test_padded)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
'''

# define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    # basic DNN
    # tf.keras.layers.GlobalAveragePooling1D(),

    # lstm
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

    # Conv1D
    tf.keras.layers.Conv1D(128, 5, activation=tf.nn.relu),
    tf.keras.layers.GlobalAveragePooling1D(),

    # GRU
    # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),

    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=EPOCHS,
                    validation_data=test_dataset)
# plotting
import matplotlib.pyplot as plt

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
