# -*- coding: utf-8 -*-
"""Device function"""

import os
import warnings
import tensorflow as tf


def set_gpu():
    gpu_list = tf.config.list_physical_devices('GPU')
    print('The version of tensorflow is: \n', tf.__version__)
    print('List of gpu devices: \n', gpu_list)

    if gpu_list:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        warnings.warn("GPU has not been recognized!")
