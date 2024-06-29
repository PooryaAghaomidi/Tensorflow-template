# -*- coding: utf-8 -*-
"""Adam loss function"""

from tensorflow.keras.optimizers import Adam


def adam_opt(lr, epsilon=1e-7, clipvalue=None):
    return Adam(learning_rate=lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=epsilon,
                amsgrad=False,
                clipnorm=None,
                clipvalue=clipvalue,
                global_clipnorm=None,
                name="adam")
