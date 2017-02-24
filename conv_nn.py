from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint
import os

import numpy as np
import pandas as pd

from keras import backend as K
K.set_image_dim_ordering('th')

if __name__ == "__main__":
    print(' ')
    print('-'*50)
    print('''
  /\/\   ___| | __ _ _ __   ___  _ __ ___   __ _  /\ \ \/\ \ \
 /    \ / _ \ |/ _` | '_ \ / _ \| '_ ` _ \ / _` |/  \/ /  \/ /
/ /\/\ \  __/ | (_| | | | | (_) | | | | | | (_| / /\  / /\  /
\/    \/\___|_|\__,_|_| |_|\___/|_| |_| |_|\__,_\_\ \/\_\ \/
        ''')
    print('Training complete model')
    print('-'*50)

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            horizontal_flip=True,
            vertical_flip=True)

    train_generator=train_datagen.flow_from_directory(
            'data/melanoma_preprocessed/',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical')

    test_datagen=ImageDataGenerator(rescale=1./255)

    test_generator=test_datagen.flow_from_directory(
            'data/Evaluation_melanoma/',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical')

    # let's train the model using SGD + momentum (how original).
    model = Sequential()

    model.add(Convolution2D(32, 3, 3,input_shape=(3, 224, 224), border_mode='full', activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='full', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='full', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3, activation='softmax'))

    print('-'*50)
    print (
    ''' Compiling model with:
    - stochastic gradient descend optimizer;
    - learning rate=0.001;
    - decay=1e-6;
    - momentum=0.9 ''')

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


    if not os.path.exists('weights/complete_model_checkpoint_weights/'):
        os.makedirs('weights/complete_model_checkpoint_weights/')
        print('Directory "weights/complete_model_checkpoint_weights/" has been created')

    filepath="weights/weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]


    epochs=int(input('How much epochs we need?:'))

    model.fit_generator(
            train_generator,
            validation_data=test_generator,
            samples_per_epoch=800,
            nb_val_samples=320,
            nb_epoch=epochs,
            class_weight='auto',
            callbacks=callbacks_list,
            verbose=1)

    print('-'*50)
    print('Training the model has been completed')
