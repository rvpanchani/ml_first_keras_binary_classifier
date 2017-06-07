from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model
import keras

def mini_conv():
    img_rows, img_cols = 150,150
    if K.image_data_format() == 'channels_first':
        input_shape = (3,img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 3)

    model = Sequential()

    conv1 = Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape)
    model.add(conv1)

    # conv2 = Conv2D(64, (3, 3), activation='relu')
    # model.add(conv2)

    maxpool1 = MaxPooling2D(pool_size=(2, 2))
    model.add(maxpool1)

    model.add(Dropout(0.25))
    model.add(Flatten())

    #model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',#keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    plot_model(model, to_file='model.png')
    return model