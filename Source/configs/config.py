# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "train_path": "../Dataset/mnist_train.csv",
        "test_path": "../Dataset/mnist_test.csv",
        "validation_path": "../Dataset/mnist_validation.csv",
        "shape": (28, 28, 1)
    },

    "train": {
        "batch_size": 32,
        "num_epochs": 10,
        "cls_num": 10,
        "learning_rate": 0.0001,
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "label_smoothing": 0.0,
        "metrics": ["accuracy"],
        "monitor": "val_loss",
        "mode": "min",
    },

    "test": {
        "model_path": "None"
    }
}
