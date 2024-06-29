# -*- coding: utf-8 -*-
"""Calculate batch numbers"""

import numpy as np


def batches_per_epoch(data_path, batch_size):
    x = np.load(data_path)
    return np.shape(x)[0] // batch_size
