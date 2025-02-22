import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Tuple, Dict

# Trainer class handles the training loop and loss computation for the Transformer model
class Trainer:
    def __init__(self, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer):
        self.model = model
        self.optimizer = optimizer
        # Track mean loss across batches
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        # Track accuracy for sequence prediction
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
    
    @tf.function  # Compile function for faster execution
    def train_step(self, inp: tf.Tensor, tar: tf.Tensor) -> Dict[str, tf.Tensor]:
        # Teacher forcing: Feed correct tokens as decoder input during training and split target into input and expected output
        tar_inp = tar[:, :-1]  # All tokens except last as input
        tar_real = tar[:, 1:]  # All tokens except first as expected output
        
        # Compute gradients and update model
        with tf.GradientTape() as tape:
            # Forward pass through the model
            predictions = self.model([inp, tar_inp], training=True)
            # Calculate loss using teacher forcing targets
            loss = self.loss_function(tar_real, predictions)
        
        # Compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)
        
        return {
            'loss': self.train_loss.result(),
            'accuracy': self.train_accuracy.result()
        }
    
    def loss_function(self, real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
        # Create mask to ignore padding tokens (zeros) in loss calculation
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        
        # Calculate cross entropy loss
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)
        
        # Apply mask to loss
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        # Return mean loss over non-padding tokens
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)