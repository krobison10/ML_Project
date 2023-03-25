#
# Kyler Robison
#
# Code for various CNN architectures.
#

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Input, Concatenate
from keras.regularizers import l2
from keras.applications import ResNet50


# Winning architecture
def big_sequential_gender():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(8192, activation='relu', kernel_regularizer=l2(0.02)))
    model.add(Dropout(0.7))
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.02)))
    model.add(Dropout(0.7))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def big_forked_gender():
    # main branch
    inputs = Input(shape=(96, 96, 1))

    main = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    main = BatchNormalization()(main)

    # forks
    # left
    left = MaxPooling2D(pool_size=(3, 3))(main)
    left = Dropout(0.5)(left)

    left = Conv2D(32, (3, 3), activation='relu', padding='same')(left)
    left = BatchNormalization()(left)

    left = MaxPooling2D(pool_size=(2, 2))(left)
    left = Dropout(0.5)(left)

    left = Conv2D(64, (3, 3), activation='relu', padding='same')(left)
    left = BatchNormalization()(left)

    left = MaxPooling2D(pool_size=(2, 2))(left)
    left = Dropout(0.5)(left)

    # right
    right = MaxPooling2D(pool_size=(3, 3))(main)
    right = Dropout(0.5)(right)

    right = Conv2D(32, (3, 3), activation='relu', padding='same')(right)
    right = BatchNormalization()(right)

    right = MaxPooling2D(pool_size=(2, 2))(right)
    right = Dropout(0.5)(right)

    right = Conv2D(64, (3, 3), activation='relu', padding='same')(right)
    right = BatchNormalization()(right)

    right = MaxPooling2D(pool_size=(2, 2))(right)
    right = Dropout(0.5)(right)

    outputs = Concatenate()([left, right])
    outputs = Flatten()(outputs)

    outputs = Dense(8192, activation='relu', kernel_regularizer=l2(0.01))(outputs)
    outputs = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(outputs)
    outputs = Dropout(0.7)(outputs)
    outputs = Dense(2, activation='sigmoid')(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# black and white, maybe try 0.2 dropouts for the conv layers
def standard_sequential_gender():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(8192, activation='relu', kernel_regularizer=l2(0.02)))
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.02)))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# best so far: filters 32, 64, dropouts 0.5, 0.5
def simple_sequential_gender():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(8192, activation='relu', kernel_regularizer=l2(0.015)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.015)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def simple_sequential_age():
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=(96, 96, 3), activation='relu', padding='same'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def big_sequential_age():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(8192, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
