# -*- coding: utf-8 -*-
"""Seed function"""

import random
import numpy as np
import tensorflow as tf


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
