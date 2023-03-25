#
# Kyler Robison
#
# Closely adapted version of the script for gender.
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


print("\n--------Training neural network for age classification using images--------\n")

data = pd.read_csv("../data/tcss455/training/profile/profile.csv", index_col=0)
data = data.loc[:, ['userid', 'age']]


# Map ages to categories
def map_fn(cell):
    age = int(cell)
    if age <= 24:
        return 0
    if age <= 34:
        return 1
    if age <= 49:
        return 2
    else:
        return 3


data['age'] = data['age'].apply(map_fn)

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

# Balance data
counts = data['age'].value_counts()
min_count = counts.min()

balanced_data = pd.DataFrame(columns=['age', 'image'])

for class_name in counts.index:
    class_df = data[data['age'] == class_name].sample(min_count)
    balanced_data = pd.concat([balanced_data, class_df], axis=0)

balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)

X = balanced_data['image'].to_numpy()
y = balanced_data['age'].to_numpy()

class_counts = data['age'].value_counts()
max_count = class_counts.max()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=800)

# convert these arrays of 3d arrays into a single 4d array each
X_train = np.stack(X_train, axis=0)
X_test = np.stack(X_test, axis=0)

# turn into categories
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)


class_weights = {0: 1.,
                 1: max_count / class_counts[0],
                 2: max_count / class_counts[1],
                 3: max_count / class_counts[2],
                 4: max_count / class_counts[3]}

cnn_model = models.big_sequential_age()
early_stopping = EarlyStopping(monitor='val_loss', patience=50)

# checkpoint to save best run as final
best_weights_path = "../models/age_cnn_best.hdf5"
checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', save_best_only=True, mode='min')

callbacks_list = [checkpoint]
history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), class_weight=class_weights,
                        epochs=80, batch_size=200, callbacks=callbacks_list)


# print evaluations
final_scores = cnn_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.4f" % final_scores[1])

best_scores_model = load_model(best_weights_path)
best_scores = best_scores_model.evaluate(X_test, y_test, verbose=0)
print("Best run accuracy: %.4f" % best_scores[1])


cnn_model.save('age_cnn.hdf5')


# -------------------------------- Evaluate -------------------------------------


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("age_accuracy_plot.png")

# summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig("age_loss_plot.png")
