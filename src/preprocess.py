import tensorflow as tf
import numpy as np

# Preprocessor handles text tokenization and sequence preparation for both source and target languages
class Preprocessor:
    def __init__(self, max_length: int = 40, vocab_size: int = 8000):
        self.max_length = max_length  # Maximum sequence length
        self.vocab_size = vocab_size  # Size of vocabulary for each language
        
        # Initialize tokenizers for source and target languages and Filter out punctuation and special characters
        self.tokenizer_src = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, 
                                                                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        self.tokenizer_tgt = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                                                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    
    def build_tokenizers(self, src_texts: list, tgt_texts: list) -> None:
        """Build vocabulary from training texts for both languages"""
        # Handle byte strings (common in TensorFlow datasets)
        src_texts = [str(text.decode('utf-8') if isinstance(text, bytes) else text) for text in src_texts]
        tgt_texts = [str(text.decode('utf-8') if isinstance(text, bytes) else text) for text in tgt_texts]
        
        # Fit tokenizers to create vocabulary mappings
        self.tokenizer_src.fit_on_texts(src_texts)
        self.tokenizer_tgt.fit_on_texts(tgt_texts)
    
    def encode(self, src_text, tgt_text):
        """Convert text to integer sequences for model input"""
        # Handle TensorFlow tensors and convert to strings
        if hasattr(src_text, 'numpy'):
            src_str = src_text.numpy().decode('utf-8')
        else:
            src_str = str(src_text)
            
        if hasattr(tgt_text, 'numpy'):
            tgt_str = tgt_text.numpy().decode('utf-8')
        else:
            tgt_str = str(tgt_text)
            
        # Convert texts to integer sequences
        src_seq = self.tokenizer_src.texts_to_sequences([src_str])[0]
        tgt_seq = self.tokenizer_tgt.texts_to_sequences([tgt_str])[0]
        
        # Add special tokens: START token (vocab_size) and END token (vocab_size + 1)
        src_seq = [self.vocab_size] + src_seq + [self.vocab_size + 1]
        tgt_seq = [self.vocab_size] + tgt_seq + [self.vocab_size + 1]
        
        # Pad sequences to fixed length for batch processing
        src_seq = tf.keras.preprocessing.sequence.pad_sequences(
            [src_seq], maxlen=self.max_length, padding='post')[0]
        tgt_seq = tf.keras.preprocessing.sequence.pad_sequences(
            [tgt_seq], maxlen=self.max_length, padding='post')[0]
        
        # Convert to int64 tensors for TensorFlow processing
        return tf.cast(src_seq, tf.int64), tf.cast(tgt_seq, tf.int64)
    
    def decode(self, seq):
        """Convert integer sequence back to text"""
        # Remove padding and special tokens before decoding
        seq = [i for i in seq if i < self.vocab_size]
        # Convert sequence back to text using target language tokenizer
        return ' '.join(self.tokenizer_tgt.sequences_to_texts([seq]))