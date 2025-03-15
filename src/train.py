import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Tuple, Dict

class Trainer:
    def __init__(self, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
    
    @tf.function
    def train_step(self, inp: tf.Tensor, tar: tf.Tensor) -> Dict[str, tf.Tensor]:
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        with tf.GradientTape() as tape:
            predictions = self.model([inp, tar_inp], training=True)
            loss = self.loss_function(tar_real, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)
        
        return {
            'loss': self.train_loss.result(),
            'accuracy': self.train_accuracy.result()
        }
    
    def loss_function(self, real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)
        
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
