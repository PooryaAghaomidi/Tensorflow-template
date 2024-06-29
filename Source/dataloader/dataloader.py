# -*- coding: utf-8 -*-
"""Data Loader"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, address, shape, batch_size, cls_num, shuffle=True):
        self.address = address
        self.data = np.load(address)
        self.shape = shape
        self.batch_size = batch_size
        self.cls_num = cls_num
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.data[k] for k in indexes]
        x, y = self.__data_generation(list_IDs_temp)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        x = np.empty((self.batch_size, int(self.shape[0]), int(self.shape[1]), int(self.shape[2])))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            x[i, :, :, 0] = ID[1:].reshape((int(self.shape[0]), int(self.shape[1])))
            y[i] = ID[0]

        return x, to_categorical(y, num_classes=self.cls_num)
