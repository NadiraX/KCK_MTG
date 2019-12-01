import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os.path
from pathlib import Path
from PIL import Image
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

dane = []
train_labels = []
train_images = []
test_images = []
test_labels = []
path = Path('pure_rotated_resize/')
x = []
for filename in os.listdir(path):
    img = Image.open(os.path.join(path,Path(filename)))
    img.load()
    data = np.asarray(img, dtype="int32")
    train_images.append(data)
    train_labels.append(filename[0])
train_images = np.array(train_images, dtype="int32")
print(train_images.shape)
print(train_labels)

for i in range(len(train_images)):
    train_images[i] = train_images[i] / 255.0
for i in range(len(test_images)):
    test_images[i] = test_images[i] / 255.0

for i in range(len(train_labels)):
    if train_labels[i]=='b':
        train_labels[i]=[0]
    if train_labels[i]=='g':
        train_labels[i]=[1]
    if train_labels[i]=='r':
        train_labels[i]=[2]
    if train_labels[i]=='w':
        train_labels[i]=[3]
    if train_labels[i]=='u':
        train_labels[i]=[4]
for i in range(len(test_labels)):
    if test_labels[i]=='b':
        test_labels[i]=[0]
    if test_labels[i]=='g':
        test_labels[i]=[1]
    if test_labels[i]=='r':
        test_labels[i]=[2]
    if test_labels[i]=='w':
        test_labels[i]=[3]
    if test_labels[i]=='u':
        test_labels[i]=[4]
train_labels=np.array(train_labels)
test_labels=np.array(test_labels)
# train_images = np.expand_dims(train_images, -1)
# test_images = np.expand_dims(test_images, -1)

#train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
# for i in range(len(train_images)):
#     train_images[i]=train_images[i].reshape(95,32, 32,3)


model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3) ))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))
#
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50)

plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

train_loss, train_acc = model.evaluate(train_images,  train_labels, verbose=2)