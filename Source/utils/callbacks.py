# -*- coding: utf-8 -*-
"""Callbacks"""

from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from livelossplot.inputs.tf_keras import PlotLossesCallback


def callback(mymonitor='val_loss', mymode='min'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    main_chk = ModelCheckpoint(filepath='checkpoints/model_{}'.format(timestamp), monitor=mymonitor, mode=mymode, verbose=1, save_best_only=True)
    early_st = EarlyStopping(monitor=mymonitor, mode=mymode, patience=30, verbose=1)
    rduce_lr = ReduceLROnPlateau(monitor=mymonitor, mode=mymode, factor=0.5, patience=5, verbose=1, min_lr=0.000001)
    tsboard = TensorBoard(log_dir='checkpoints/tensorboard_{}'.format(timestamp))

    tr_plot = PlotLossesCallback()

    return [main_chk, early_st, rduce_lr, tr_plot, tsboard], 'checkpoints/model_{}'.format(timestamp)
