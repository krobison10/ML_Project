#
# Kyler Robison
#
# A looper for testing face detection cases.
#

import pandas as pd
import cv2

print("\n\nTraining a neural network for gender classification using images\n")

data = pd.read_csv("../data/tcss455/training/profile/profile.csv", index_col=0)
data = data.loc[:, ['userid', 'gender']]

all_uids = data['userid']

print("Loading images")

# add images column
data['image'] = ''

counter = 0
for uid in all_uids:
    image = cv2.imread('data/tcss455/training/image/' + uid + '.jpg')
    data.at[counter, 'image'] = image
    counter += 1

print("Finished\n\n")


# print(data.loc[0, :]) # access a row of an index
# data.at[0, 'image'] = 'Banana' # modify cell
# data = data.drop(index=0) # drop a row of an index

print("Preprocessing images..")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# for i in range(0, data.shape[0]):
for i in range(0, 50):
    image = data.at[i, 'image']
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)

    # if not one face, throw out
    if len(faces) != 1:
        print("Error, # faces = %i" % len(faces))
        cv2.imshow("Error", image)
        cv2.waitKey()
        data = data.drop(index=i)
        continue

    for (x, y, w, h) in faces:
        im_selection = image[y:y + h, x:x + w]
        im_selection = cv2.resize(im_selection, (48, 48), interpolation=cv2.INTER_LANCZOS4)
        cv2.imshow("face", im_selection)

    cv2.waitKey()

print("Finished")


