import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random
import numpy as np
import os.path
from pathlib import Path
from PIL import Image
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return(a,b)
dane = []
train_labels = []
train_images = []
test_images = []
test_labels = []
path = Path('pure_rotated_resize/')
path_train = Path('train_resize/')

x = []
for filename in os.listdir(path):
    img = Image.open(os.path.join(path,Path(filename))).convert("RGB")
    img.load()
    data = np.asarray(img, dtype="int32")
    train_images.append(data)
    train_labels.append(filename[0])

for filename in os.listdir(path_train):
    img = Image.open(os.path.join(path_train,Path(filename))).convert("RGB")
    img.load()
    data = np.asarray(img, dtype="int32")
    train_images.append(data)
    train_labels.append(filename[0])

#for filename in os.listdir(path_train):


train_images = np.array(train_images, dtype="int32")
test_images = np.array(test_images, dtype="int32")


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
    elif train_labels[i]=='a':
        train_labels[i]=[5]
    elif train_labels[i]=='m':
        train_labels[i]=[6]
for i in range(len(test_labels)):
    if test_labels[i]=='b':
        test_labels[i]=[0]
    elif test_labels[i]=='g':
        test_labels[i]=[1]
    elif test_labels[i]=='r':
        test_labels[i]=[2]
    elif test_labels[i]=='w':
        test_labels[i]=[3]
    elif test_labels[i]=='u':
        test_labels[i]=[4]
    elif test_labels[i]=='a':
        test_labels[i]=[5]
    elif test_labels[i]=='m':
        test_labels[i]=[6]

# c = list(zip(train_labels,train_images))
# random.shuffle(c)
# train_labels,train_images = zip(*c)

train_labels=np.array(train_labels)
test_labels=np.array(test_labels)

train_images,train_labels = shuffle_in_unison_scary(train_images,train_labels)
#train_images,train_labels = shuffle_in_unison_scary(train_images,train_labels)
#print(train_labels)
# train_images = np.expand_dims(train_images, -1)
# test_images = np.expand_dims(test_images, -1)

#train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
# for i in range(len(train_images)):
#     train_images[i]=train_images[i].reshape(95,32, 32,3)


model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3) ))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(125, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
#
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,validation_split=0.30)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#
# print(test_acc)