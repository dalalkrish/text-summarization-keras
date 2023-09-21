# import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import datetime
# from tensorflow.keras.text.tokenize import Tokenizer
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, TimeDistributed, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras import Model
from tensorflow.keras.utils import pad_sequences, to_categorical
import os
import pickle

log_dir = "logs/trial_fit_1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

### Write generator to read text files of News Articles and Summaries
def read_files(file_names):
    """
    Function to read the text file and yield the text.
    It only supports text files with .txt extension
    Argument:
        - file_name: Iterator containing file names
    """
    articles = []
    summaries = []
    for file_name in file_names:
        try:
            if file_name.endswith('.txt'):
                txt = open(file_name, 'r').read()
                if file_name.split('/')[2]=='News Articles':
                    articles.append(txt)
                elif file_name.split('/')[2]=='Summaries':
                    summaries.append(txt)
                else:
                    pass
            else:
                raise "Only supports text files"
        except Exception as e:
            print(f"{file_name} returned an Exception: {e}")

    return np.array(articles), np.array(summaries)

def file_path_generator():
    """
    File path generator function
    """
    for root, subdirs, files in os.walk('./BBC News Summary/', topdown=True):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                # print(file_path)
                yield file_path

def tokenize_and_pad(source, texts, num_words=5000, max_len=None, pad='post'):
    """
    Function to train the tokenizer and pad the sequences to even length
    Arguments:
     - source: label or target to append special tokens
     - texts: Iterator containing texts
     - max_len: lenght to which the sequences should be padded
     - pad: post or pre padding. It applies to truncating argument too
    """
    if source=='target':
        texts = ['<START> '+t+ ' <END>' for t in texts]

    tokenizer = Tokenizer(num)

    tokenizer.fit_on_texts(texts)
    seq = tokenizer.texts_to_sequences(texts)

    if max_len is not None:
        seq = pad_sequences(seq, maxlen=max_len, padding=pad, truncating=pad)
    else:
        seq = pad_sequences(seq, padding=pad, truncating=pad)

    return tokenizer, seq

def tf_lower_and_split_punct(text):
  # Split accented characters.
  text = tf_text.normalize_utf8(text, 'NFKD')
  text = tf.strings.lower(text)
  # Keep space, a to z, and select punctuation.
  text = tf.strings.regex_replace(text, '[^ 0-9a-z.?!,¿]', '')
  # Add spaces around punctuation.
  text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
  # Strip whitespace.
  text = tf.strings.strip(text)

  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text

articles, summaries = read_files(file_path_generator())

BUFFER_SIZE = len(articles)
BATCH_SIZE = 64
EPOCHS = 100
MAX_VOCAB_SIZE = 5000

is_train = np.random.uniform(size=len(articles))<0.8

train_raw = (tf.data.Dataset
             .from_tensor_slices((articles[is_train], summaries[is_train]))
             .shuffle(BUFFER_SIZE)
             .batch(BATCH_SIZE))

val_raw = (tf.data.Dataset
           .from_tensor_slices((articles[~is_train], summaries[~is_train]))
           .shuffle(BUFFER_SIZE)
           .batch(BATCH_SIZE))

context_vectorizer = tf.keras.layers.TextVectorization(
    standardize = tf_lower_and_split_punct,
    max_tokens = MAX_VOCAB_SIZE,
    ragged=True)

target_vectorizer = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=MAX_VOCAB_SIZE,
    ragged=True)

context_vectorizer.adapt(train_raw.map(lambda context, target: context))
target_vectorizer.adapt(train_raw.map(lambda context, target: target))

def store_objects(obj, fname):
    fpath = './obj_store_trial_1/'
    pickle.dump({'config': obj.get_config(),
                 'weights': obj.get_weights()}
                , open(fpath+fname+".pkl", "wb"))

def preprocess_text(context, target):
    context = context_vectorizer(context).to_tensor()
    target = target_vectorizer(target)

    target_in  = target[:,:-1].to_tensor()
    target_out = target[:,1:].to_tensor()

    return (context, target_in), target_out

train_ds = train_raw.map(preprocess_text, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

val_ds = val_raw.map(preprocess_text, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

def encoder(hsize, embed_dim=200):
    en_input_layer = Input(shape=(None,), name='encoder_input_layer', ragged=True)
    en_embed = Embedding(context_vectorizer.vocabulary_size(), output_dim=embed_dim, name='encoder_embedding_layer')
    en_embed_out = en_embed(en_input_layer)
    en_gru_1 = GRU(hsize, return_sequences=True, return_state=True, name='encoder_gru_layer_1')
    en_gru_1_out, en_gru_states = en_gru_1(en_embed_out)
    return en_input_layer, en_gru_1_out, en_gru_states

def decoder(hsize, encoder_inputs, encoder_states, embed_dim=200):
    de_input_layer = Input(shape=(None,), name='decoder_input_layer', ragged=True)

    ### Decoder Embedding Layer to look up the Target sequence
    de_embed = Embedding(target_vectorizer.vocabulary_size(), output_dim=embed_dim, name='decoder_embedding_layer')
    de_embed_out = de_embed(de_input_layer)

    ### Decoder RNN to keep track of the last Token that was generated to predict the next one
    de_gru_1 = GRU(hsize, return_sequences=True, name='decoder_gru_layer_1')
    de_gru_1_out = de_gru_1(de_embed_out, initial_state=encoder_states)

    ### Encoder weights to attend to while predicting the next token by looking up the current predicted token
    de_attention = MultiHeadAttention(num_heads=1, key_dim=hsize, name='multi_head_attention_layer')
    de_attention_out, attention_weights = de_attention(query=de_gru_1_out, value=encoder_inputs, return_attention_scores=True)

    # attention_weights = tf.reduce_mean(attention_weights, axis=1)

    # de_attention_out = Add(name='add_attention_weights_layer')([de_gru_1_out, de_attention_out])
    de_gru_1_out = LayerNormalization(name='attention_norm_layer')(de_attention_out)

    ### Final Time Distributed Dense layer to produce the logits for the Target Vocabulary
    de_dense = TimeDistributed(Dense(target_vectorizer.vocabulary_size(), activation='softmax'), name='time_distributed_output_layer')
    de_preds = de_dense(de_gru_1_out)
    return de_input_layer, de_preds, attention_weights

hsize = 256

def create_model(hsize):
    en_input_layer, enc_out, enc_states = encoder(hsize)
    de_input_layer, de_preds, attention_weights = decoder(hsize, enc_out, enc_states)
    model = Model([en_input_layer, de_input_layer], de_preds)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                    metrics=["acc"])
    return model, attention_weights

if __name__=='__main__':
    m, attention_weights = create_model(hsize)

    vocab_size = 1.0 * target_vectorizer.vocabulary_size()
    m.summary()

    print(f"Expected loss: {tf.math.log(vocab_size).numpy():.4f}\nExpected accuracy: {(1/vocab_size):.4f}")
    m.evaluate(val_ds.repeat(20), steps=20, return_dict=True)

    print('\nTraining Starts:')
    print('='*50)

    history = m.fit(
        train_ds,
        epochs=100,
        # steps_per_epoch=EPOCHS,
        validation_data=val_ds,
        # validation_steps=20,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('./checkpoints_trial_1/',
                                               save_best_only=True,
                                               save_weights_only=False),
            tf.keras.callbacks.EarlyStopping(patience=10),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)])

    store_objects(context_vectorizer, 'context_vectorizer')
    store_objects(target_vectorizer, 'target_vectorizer')
