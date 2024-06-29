# -*- coding: utf-8 -*-
""" main.py """

import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from model.model import build
from configs.config import CFG
from utils.set_seed import set_seed
from utils.set_device import set_gpu
from utils.callbacks import callback
from utils.per_epoch import batches_per_epoch
from train.train import TrainModel
from test.test_model import test_model
from dataloader.dataloader import DataGenerator
from dataset.preprocessing import norm_data


def run(train_par, data_par, test_par, mode):
    set_seed()
    set_gpu()

    datasets = os.listdir('dataset/')
    if ('train.npy' in datasets) and ('test.npy' in datasets) and ('val.npy' in datasets):
        pass
    else:
        norm_data(data_par['train_path'], data_par['test_path'], data_par['validation_path'])

    train_gen = DataGenerator('dataset/train.npy', data_par['shape'], train_par['batch_size'], train_par['cls_num'],
                              shuffle=False)
    test_gen = DataGenerator('dataset/test.npy', data_par['shape'], train_par['batch_size'], train_par['cls_num'],
                             shuffle=False)
    val_gen = DataGenerator('dataset/val.npy', data_par['shape'], train_par['batch_size'], train_par['cls_num'],
                            shuffle=False)

    model = build(data_par['shape'], train_par['cls_num'])
    callbacks, model_name = callback(train_par['monitor'], train_par['mode'])
    train_per_epoch = batches_per_epoch('dataset/train.npy', train_par['batch_size'])
    test_per_epoch = batches_per_epoch('dataset/test.npy', train_par['batch_size'])
    val_per_epoch = batches_per_epoch('dataset/val.npy', train_par['batch_size'])

    if train_par['loss'] == 'categorical_crossentropy':
        from loss.categorical_crossentropy import cc_loss
        my_loss = cc_loss(from_logits=False, label_smoothing=train_par['label_smoothing'])
    else:
        warnings.warn("The loss is invalid")

    if train_par['optimizer'] == 'adam':
        from optimizer.adam import adam_opt
        my_opt = adam_opt(train_par['learning_rate'], epsilon=1e-7, clipvalue=None)
    else:
        warnings.warn("The optimizer is invalid")

    train_class = TrainModel(model, callbacks, my_loss, my_opt, train_par['metrics'], train_par['num_epochs'],
                             train_par['batch_size'], train_gen, val_gen, train_per_epoch, val_per_epoch)

    if mode == 'train':
        train_class.train()
        los, acc = test_model(model_name, test_gen, test_per_epoch)
        print('The final loss is: ', los)
        print('The final accuracy is: ', acc)

    elif mode == 'test':
        los, acc = test_model(test_par['model_path'], test_gen, test_per_epoch)
        print('The final loss is: ', los)
        print('The final accuracy is: ', acc)

    else:
        warnings.warn("The mode is invalid")


if __name__ == '__main__':
    mode = 'train'
    train_par = CFG['train']
    data_par = CFG['data']
    test_par = CFG['test']

    run(train_par, data_par, test_par, mode)
