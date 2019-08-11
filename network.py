import tensorflow as tf
import os

def scriptPath():
    return os.path.realpath(__file__)

def convUnit(layer,num_filters):
    x = tf.keras.layers.Conv2D(filters = num_filters,kernel_size = (3,3),padding='SAME')(layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def transposeConvUnit(layer,num_filters):
    x = tf.keras.layers.Conv2DTranspose(filters = num_filters,kernel_size = (3,3),strides=2,padding='SAME')(layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def UNet(input_shape):
    input_tensor = tf.keras.layers.Input(shape = input_shape)
    c1 = convUnit(input_tensor,32)
    m1 = tf.keras.layers.MaxPool2D()(c1)

    c2 = convUnit(m1,32)
    m2 = tf.keras.layers.MaxPool2D()(c2)

    c3 = convUnit(m2,64)
    m3 = tf.keras.layers.MaxPool2D()(c3)

    c4 = convUnit(m3,64)
    m4 = tf.keras.layers.MaxPool2D()(c4)

    c5 = convUnit(m4,128)
    drop5 = tf.keras.layers.Dropout(rate = 0.3)(c5)

    upc6 = transposeConvUnit(drop5,64)
    c6 = convUnit(upc6,64)
    add1 = tf.keras.layers.Concatenate()([c4,c6])

    upc7 = transposeConvUnit(add1,64)
    c7 = convUnit(upc7,64)
    add2 = tf.keras.layers.Concatenate()([c3,c7])

    upc8 = transposeConvUnit(add2,32)
    c8 = convUnit(upc8,32)
    add3 = tf.keras.layers.Concatenate()([c2,c8])
    
    upc9 = transposeConvUnit(add3,32)
    c9 = convUnit(upc9,32)
    out = tf.keras.layers.Conv2D(filters = 3,kernel_size = (3,3),padding='SAME',activation='softmax')(c9)

    model = tf.keras.models.Model(inputs=input_tensor,outputs = out)
    return model
