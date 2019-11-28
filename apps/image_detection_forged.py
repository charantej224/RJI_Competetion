import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

from pylab import *
from PIL import Image, ImageChops, ImageEnhance

import itertools
import os
import matplotlib.pyplot as plot

sns.set(style='white', context='notebook', palette='deep')


def remove_saved():
    labels = os.listdir('real-and-fake-face-detection/')
    for label in labels:
        pics = os.listdir('real-and-fake-face-detection/{}/'.format(label))
        for pic in pics:
            if "renamed" in pic:
                os.remove('real-and-fake-face-detection/{}/{}'.format(label, pic))
    print("files removed. ")


def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    ELA_filename = filename.split('.')[0] + '.ela.png'
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    return ela_im


labels = os.listdir('real-and-fake-face-detection/')

x_data = []
y_data = []

for label in labels:
    pics = os.listdir('real-and-fake-face-detection/{}/'.format(label))
    for pic in pics:
        x_data.append(array(convert_to_ela_image('real-and-fake-face-detection/{}/{}'.format(label, pic), 90).resize(
            (128, 128))).flatten() / 255.0)
        y_data.append(label)

x_data = np.array(x_data)
y_data = np.array(y_data)

x_data = x_data.reshape(-1, 128, 128, 3)

enc = LabelEncoder().fit(y_data)
y_encoded = enc.transform(y_data)
y_data = to_categorical(y_encoded)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=5)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid',
                 activation='relu', input_shape=(128, 128, 3)))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid',
                 activation='relu'))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

model.summary()

optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=2, verbose=0, mode='auto')

epochs = 2
batch_size = 100

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping])

fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks = np.arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=45)
    plot.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, cm[i, j],
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plot.tight_layout()
    plot.ylabel('True label')
    plot.xlabel('Predicted label')


# Predict the values from the validation dataset
y_pred = model.predict(x_val)
# Convert predictions classes to one hot vectors
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(2))

# remove created files.
remove_saved()
