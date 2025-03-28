import tensorflow as tf
from preprocess import Preprocessor

class Translator:
    def __init__(self, model: tf.keras.Model, preprocessor: Preprocessor):
        self.model = model
        self.preprocessor = preprocessor
    
    def translate(self, sentence: str) -> str:
        input_tokens, _ = self.preprocessor.encode(sentence, "")
        encoder_input = tf.expand_dims(input_tokens, 0)
        
        decoder_input = tf.constant([self.preprocessor.vocab_size], dtype=tf.int32)
        output = tf.expand_dims(decoder_input, 0)
        
        for _ in range(self.preprocessor.max_length):
            predictions = self.model([encoder_input, output], training=False)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), dtype=tf.int32)
            
            if int(predicted_id[0][0]) == self.preprocessor.vocab_size + 1:
                break
                
            output = tf.concat([output, predicted_id], axis=-1)
        
        output_sequence = [int(i) for i in output[0].numpy() if int(i) < self.preprocessor.vocab_size]
        text = self.preprocessor.decode(output_sequence)
        
        return text
