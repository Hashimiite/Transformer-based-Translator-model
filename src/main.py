import tensorflow as tf
import tensorflow_datasets as tfds
from preprocess import Preprocessor
from model import Transformer
from train import Trainer
from translate import Translator

def load_dataset():
    examples, metadata = tfds.load('wmt14_translate/fr-en', with_info=True,
                                 as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']
    
    train_examples = train_examples.take(60000)
    val_examples = val_examples.take(2000)
    
    return train_examples, val_examples

def main():
    train_examples, val_examples = load_dataset()
    
    preprocessor = Preprocessor(max_length=50, vocab_size=16000) 
    
    train_src = [ex[0].numpy().decode('utf-8') for ex in train_examples]
    train_tgt = [ex[1].numpy().decode('utf-8') for ex in train_examples]
    preprocessor.build_tokenizers(train_src, train_tgt)
    
    def prepare_data(src, tgt):
        src_tensor, tgt_tensor = tf.py_function(
            preprocessor.encode, 
            [src, tgt], 
            [tf.int64, tf.int64]
        )
        src_tensor.set_shape([preprocessor.max_length])
        tgt_tensor.set_shape([preprocessor.max_length])
        return src_tensor, tgt_tensor

    train_dataset = train_examples.map(prepare_data)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(20000)  
    train_dataset = train_dataset.padded_batch(
        128,
        padded_shapes=([preprocessor.max_length], [preprocessor.max_length])
    )
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    model = Transformer(
        num_layers=6,       
        d_model=512,      
        num_heads=8,
        dff=2048,        
        input_vocab_size=preprocessor.vocab_size + 2,
        target_vocab_size=preprocessor.vocab_size + 2,
        max_position=preprocessor.max_length,
        rate=0.1
    )
    
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super().__init__()
            self.d_model = tf.cast(d_model, tf.float32)
            self.warmup_steps = warmup_steps
            
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    learning_rate = CustomSchedule(512)
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    trainer = Trainer(model, optimizer)
    
    print("Starting training...")
    epochs = 20 
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch, (inp, tar) in enumerate(train_dataset):
            trainer_metrics = trainer.train_step(inp, tar)
            if batch % 50 == 0:
                print(f'Batch {batch} Loss: {trainer_metrics["loss"]:.4f} '
                      f'Accuracy: {trainer_metrics["accuracy"]:.4f}')
        
        if (epoch + 1) % 5 == 0:
            test_sentence = "Hello, how are you?"
            translator = Translator(model, preprocessor)
            translation = translator.translate(test_sentence)
            print(f"\nTest translation at epoch {epoch + 1}:")
            print(f"Input: {test_sentence}")
            print(f"Output: {translation}\n")
    
    model.save_weights('models/translation_model')
    
    translator = Translator(model, preprocessor)
    test_sentences = [
        "Hello, how are you?",
        "What is your name?",
        "I love learning languages.",
        "The weather is nice today."
    ]
    print("\nFinal test translations:")
    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"Input: {sentence}")
        print(f"Output: {translation}\n")

if __name__ == "__main__":
    main()
