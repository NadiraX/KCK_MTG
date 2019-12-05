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
from functools import reduce
from PIL import ImageFont

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
path_predict = Path('predict/')

x = []


for filename in os.listdir(path_train):
    img = Image.open(os.path.join(path_train,Path(filename))).convert("RGB")
    img.load()
    data = np.asarray(img, dtype="int32")
    train_images.append(data)
    train_labels.append(filename[0])
for filename in os.listdir(path):
    img = Image.open(os.path.join(path,Path(filename))).convert("RGB")
    img.load()
    data = np.asarray(img, dtype="int32")
    train_images.append(data)
    train_labels.append(filename[0])
# for filename in os.listdir(path):
#     img = Image.open(os.path.join(path,Path(filename))).convert("RGB")
#     img.load()
#     data = np.asarray(img, dtype="int32")
#     test_images.append(data)
#     test_labels.append(filename[0])
zipped = list(zip(train_images,train_labels))
t2 = [zipped.pop(random.randrange(len(zipped))) for x in range(60)]
train_images,train_labels = list(zip(*zipped))
train_images = list(train_images)
train_labels = list(train_labels)
test_images,test_labels = list(zip(*t2))
test_images = list(test_images)
test_labels = list(test_labels)
# for filename in os.listdir(path):
#     img = Image.open(os.path.join(path,Path(filename))).convert("RGB")
#     img.load()
#     data = np.asarray(img, dtype="int32")
#     test_images.append(data)
#     test_labels.append(filename[0])

#for filename in os.listdir(path_train):

print(len(train_labels))
print(len(test_labels))
train_images = np.array(train_images, dtype="float32")
test_images = np.array(test_images, dtype="float32")




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
class_names = ['b','g','r','w','u','a','m']
# c = list(zip(train_labels,train_images))
# random.shuffle(c)
# train_labels,train_images = zip(*c)

train_labels=np.array(train_labels)
test_labels=np.array(test_labels)

train_images,train_labels = shuffle_in_unison_scary(train_images,train_labels)
test_images,test_labels = shuffle_in_unison_scary(test_images,test_labels)


model = models.Sequential()
# model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(100,100,3) ))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Flatten(input_shape = (100,100,3)))
# model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2500, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
#
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,validation_split=0.2)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# print(test_acc)
# prediction = model.predict(test_images)
# print(prediction)
# print(class_names[np.argmax(prediction[0])])


for i in range(len(test_images)):
    test_images[i] = test_images[i] * 255

predict = model.predict(test_images)
all = [class_names[np.argmax(predict[i])] for i in range(len(predict))]
test_labels = test_labels.tolist()
test_labels = reduce(lambda x,y: x+y,test_labels)
print(test_labels)
all_true = [class_names[test_labels[i]] for i in range(len(test_labels))]
print(test_acc)
print(all)
print(all_true)
for i in range(len(test_images)):
    im = Image.fromarray(test_images[i].astype('uint8'))
    im = im.resize((100,100),Image.ANTIALIAS)
    filename = Path(str(i)+'.jpg')
    im.save(os.path.join(path_predict,Path(filename)))
