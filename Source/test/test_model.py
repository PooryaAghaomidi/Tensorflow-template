# -*- coding: utf-8 -*-
""" Test the model """

from tensorflow.keras.models import load_model


def test_model(model_path, test_gen, steps_per_test):
    print('Test results: ')

    testmodel = load_model(model_path)
    tst_loss, tst_acc = testmodel.evaluate(test_gen, steps=steps_per_test)
    return tst_loss, tst_acc
