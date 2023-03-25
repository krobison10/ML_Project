#
# Kyler Robison
#
# Trains the image model for gender.
#

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

import progress_bars
import models


# -------------------------------- Init --------------------------------------


print("\n--------Training neural network for gender classification using images--------\n")

data = pd.read_csv("../data/tcss455/training/profile/profile.csv", index_col=0)
data = data.loc[:, ['userid', 'gender']]

all_uids = data['userid']


# -------------------------------- Load --------------------------------------


# add images column
data['image'] = ''

counter = 0
for uid in all_uids:
    progress_bars.print_percent_done(counter, len(all_uids), title="Loading images")
    image = cv.imread('data/tcss455/training/image/' + uid + '.jpg')
    data.at[counter, 'image'] = image
    counter += 1

print("\nImage count: %d" % data.shape[0] + "\n")


# -------------------------------- Preprocess --------------------------------


face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

orig_len = data.shape[0]

for i in range(0, data.shape[0]):
    progress_bars.print_percent_done(i, orig_len, title="Preprocessing images")
    image = data.at[i, 'image']
    faces = face_cascade.detectMultiScale(cv.cvtColor(image, cv.COLOR_BGR2GRAY), 1.1, 4)

    # if not one face, throw out
    if len(faces) != 1:
        data = data.drop(index=i)
        continue

    result_image = None
    for (X, y, w, h) in faces:
        result_image = cv.resize(image[y:y + h, X:X + w], (96, 96), interpolation=cv.INTER_CUBIC).astype('float32')

    # convert to grayscale
    result_image = cv.cvtColor(result_image, cv.COLOR_BGR2GRAY)
    result_image = cv.resize(result_image, (96, 96))

    result_image /= 255

    data.at[i, 'image'] = result_image

print("\nImage count: %d" % data.shape[0] + "\n")


# -------------------------------- Train -------------------------------------


print("Training CNN...\n")

X = data['image'].to_numpy()
y = data['gender'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=7500)

# convert these arrays of 3d arrays into a single 4d array each
X_train = np.stack(X_train, axis=0)
X_test = np.stack(X_test, axis=0)

# turn into categories
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

cnn_model = models.big_sequential_gender()
early_stopping = EarlyStopping(monitor='val_loss', patience=15)

# checkpoint to save best run as final
best_weights_path = "../models/gender_cnn_best.hdf5"
checkpoint = ModelCheckpoint(best_weights_path, monitor='val_accuracy', save_best_only=True, mode='max')

callbacks_list = [checkpoint]
history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=100, batch_size=250, callbacks=callbacks_list)


# -------------------------------- Evaluate -------------------------------------


# print evaluations
final_scores = cnn_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.4f" % final_scores[1])

best_scores_model = load_model(best_weights_path)
best_scores = best_scores_model.evaluate(X_test, y_test, verbose=0)
print("Best run accuracy: %.4f" % best_scores[1])

# cnn_model.save('gender_cnn.hdf5')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("accuracy_plot.png")

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("loss_plot.png")
