# -*- coding: utf-8 -*-
""" preprocessing """

import numpy as np
import pandas as pd


def normalizer(csv_adrs):
    csv = pd.read_csv(csv_adrs).to_numpy().astype(np.float32)

    for idx, row in enumerate(csv):
        csv[idx, 1:] = (row[1:] - row[1:].min()) / (row[1:].max() - row[1:].min())

    return csv


def norm_data(train_adrs, test_adrs, val_adrs):
    train_csv = normalizer(train_adrs)
    test_csv = normalizer(test_adrs)
    val_csv = normalizer(val_adrs)

    np.save('dataset/train.npy', train_csv)
    np.save('dataset/test.npy', test_csv)
    np.save('dataset/val.npy', val_csv)
