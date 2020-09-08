# source
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%201.ipynb#scrollTo=5NEpdhb8AxID
'''
binary classification

'''
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

train_sentences = []
train_labels = []
test_sentences = []
test_labels = []
for s, l in train_data:
    train_sentences.append(s.numpy().decode('utf8'))
    train_labels.append(l.numpy())

for s, l in test_data:
    test_sentences.append(s.numpy().decode('utf8'))
    test_labels.append(l.numpy())

train_labels_final = np.array(train_labels)
test_labels_final = np.array(test_labels)

# preprocessing
EPOCHS = 20
VOCAB_SIZE = 10000
OOV_TOKEN = "oov"
MAX_LENGTH = 40
EMBEDDING_DIM = 16

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(train_padded, train_labels_final,
                    epochs=EPOCHS,
                    validation_data=(test_padded, test_labels_final))

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
print(history.history['accuracy'])
