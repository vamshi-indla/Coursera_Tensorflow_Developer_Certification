# source
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb#scrollTo=it1c0jCiNCIM

'''
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip \
    -O /tmp/rps.zip

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip \
    -O /tmp/rps-test-set.zip
'''

# initalize variabls
BATCH_SIZE = 32
EPOCHS = 25

# data preparation
import zipfile, os

train_path = '/tmp/rps.zip'
zip_ref = zipfile.ZipFile(train_path)
zip_ref.extractall('/tmp/')

validation_path = '/tmp/rps-test-set.zip'
zip_ref = zipfile.ZipFile(validation_path)
zip_ref.extractall('/tmp/')

TRAIN_DIR = os.path.join('/tmp/rps/')
rock_dir = os.path.join('/tmp/rps/rock')
paper_dir = os.path.join('/tmp/rps/paper')
scissors_dir = os.path.join('/tmp/rps/scissors')

TRAIN_SIZE = len(os.listdir(rock_dir)) + len(os.listdir(paper_dir)) + len(os.listdir(scissors_dir))

print('total training rock images: ', len(os.listdir(rock_dir)))
print('total training paper images: ', len(os.listdir(paper_dir)))
print('total training scissor images: ', len(os.listdir(scissors_dir)))

# validation
VALIDATION_DIR = os.path.join('/tmp/rps-test-set/')
rock_validation_dir = os.path.join('/tmp/rps-test-set/rock')
paper_validation_dir = os.path.join('/tmp/rps-test-set/paper')
scissors_validation_dir = os.path.join('/tmp/rps-test-set/scissors')

VALIDATION_SIZE = len(os.listdir(rock_validation_dir)) + \
                  len(os.listdir(paper_validation_dir)) + \
                  len(os.listdir(scissors_validation_dir))

print('total validation rock images: ', len(os.listdir(rock_validation_dir)))
print('total validation paper images: ', len(os.listdir(paper_validation_dir)))
print('total validation scissor images: ', len(os.listdir(scissors_validation_dir)))

# data preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1 / 255.0,
                                   width_shift_range=0.2,
                                   rotation_range=40,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest'
                                   )
validation_datagen = ImageDataGenerator(rescale=1 / 255.0)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=(150, 150),
                                                    class_mode='categorical',
                                                    batch_size=BATCH_SIZE)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size=(150, 150),
                                                              class_mode='categorical',
                                                              batch_size=BATCH_SIZE)

# define model
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(150, 150, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
history = model.fit(train_generator,
                    steps_per_epoch=int(TRAIN_SIZE / BATCH_SIZE),
                    validation_data=validation_generator,
                    validation_steps=int(VALIDATION_SIZE / VALIDATION_SIZE),
                    verbose=1,
                    epochs=EPOCHS
                    )

# plotting
import matplotlib.pyplot as plt

acc = history.history['accuracy']
loss = history.history['loss']

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training accuracy')
plt.plot(epochs_range, val_acc, label='Validation accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplots(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='lower right')
plt.title('Training and Validation Loss')
plt.show()
