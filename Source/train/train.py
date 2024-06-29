# -*- coding: utf-8 -*-
"""Train model"""


class TrainModel:
    def __init__(self,
                 model,
                 callbacks,
                 loss,
                 optimizer,
                 metrics,
                 epoches,
                 batch_size,
                 train_dataset,
                 validation_dataset,
                 steps_per_epoch,
                 validation_steps):
        self.model = model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.epoches = epoches
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

    def train(self):
        history = self.model.fit(self.train_dataset,
                                 validation_data=self.validation_dataset,
                                 batch_size=self.batch_size,
                                 epochs=self.epoches,
                                 verbose=1,
                                 callbacks=self.callbacks,
                                 steps_per_epoch=self.steps_per_epoch,
                                 validation_steps=self.validation_steps)

        return history.history['loss'], history.history['val_loss']
