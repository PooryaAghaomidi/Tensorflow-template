# -*- coding: utf-8 -*-
"""Categorical loss"""

from tensorflow.keras.losses import CategoricalCrossentropy


def cc_loss(from_logits=False, label_smoothing=0.0):
    return CategoricalCrossentropy(from_logits=from_logits,
                                   label_smoothing=label_smoothing,
                                   name="categorical_crossentropy")
