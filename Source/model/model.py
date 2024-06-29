# -*- coding: utf-8 -*-
"""Unet model"""

from tensorflow.keras import layers
from tensorflow.keras.models import Model


def build(input_shape, num_classes):
    inpt = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inpt)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)

    x = layers.Dropout(0.5)(x)
    y = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inpt, outputs=y)

    print(model.summary())

    return model
