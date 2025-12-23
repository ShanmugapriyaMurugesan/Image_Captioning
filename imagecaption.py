import time
from textwrap import wrap
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    GRU,
    Add,
    Attention,
    Dense,
    Embedding,
    LayerNormalization,
    Reshape,
    StringLookup,
    TextVectorization,
)
import gradio as gr

print("TF version:", tf.__version__)

VOCAB_SIZE = 20000
ATTENTION_DIM = 128          # reduced for memory
WORD_EMBEDDING_DIM = 128
IMG_HEIGHT = 224             # reduced for memory
IMG_WIDTH = 224
IMG_CHANNELS = 3
GCS_DIR = "gs://asl-public/data/tensorflow_datasets/"
BUFFER_SIZE = 1000
BATCH_SIZE = 8               # start small on Kaggle
MAX_CAPTION_LEN = 64


def get_image_label(example):
    caption = example["captions"]["text"][0]  # first caption per image
    img = example["image"]
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0
    return {"image_tensor": img, "caption": caption}

trainds = tfds.load("coco_captions", split="train", data_dir=GCS_DIR)
trainds = trainds.map(
    get_image_label, num_parallel_calls=tf.data.AUTOTUNE
).shuffle(BUFFER_SIZE)
trainds = trainds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Visualize a few samples
f, ax = plt.subplots(1, 4, figsize=(20, 5))
for idx, data in enumerate(trainds.take(4)):
    ax[idx].imshow(data["image_tensor"].numpy())
    caption = "\n".join(wrap(data["caption"].numpy().decode("utf-8"), 30))
    ax[idx].set_title(caption)
    ax[idx].axis("off")
plt.show()


def add_start_end_token(data):
    start = tf.convert_to_tensor("<start>")
    end = tf.convert_to_tensor("<end>")
    data["caption"] = tf.strings.join(
        [start, data["caption"], end], separator=" "
    )
    return data

trainds = trainds.map(add_start_end_token)


def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    # Remove punctuation but keep angle brackets for <start>, <end>
    return tf.strings.regex_replace(
        inputs, r"[!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]\^_`{|}~]?", ""
    )

tokenizer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    standardize=standardize,
    output_sequence_length=MAX_CAPTION_LEN,
)

tokenizer.adapt(
    trainds.map(lambda x: x["caption"]).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
)

# Vocabulary lookup
word_to_index = StringLookup(
    mask_token="", vocabulary=tokenizer.get_vocabulary()
)
index_to_word = StringLookup(
    mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True
)

def create_ds_fn(data):
    img_tensor = data["image_tensor"]
    caption = tokenizer(data["caption"])  # shape (MAX_CAPTION_LEN,)

    # Teacher forcing: input is all tokens except last, target is all tokens except first
    caption_in = caption[:-1]   # length MAX_CAPTION_LEN-1
    caption_out = caption[1:]   # length MAX_CAPTION_LEN-1

    return (img_tensor, caption_in), caption_out

batched_ds = (
    trainds.map(create_ds_fn)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

for (img, caption), label in batched_ds.take(1):
    print(f"Image shape: {img.shape}")
    print(f"Caption input shape: {caption.shape}")
    print(f"Label shape: {label.shape}")


FEATURE_EXTRACTOR = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
)
FEATURE_EXTRACTOR.trainable = False

image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image_input")
image_features = FEATURE_EXTRACTOR(image_input)
FEATURES_SHAPE = FEATURE_EXTRACTOR.output_shape[1:]  # (h, w, c)
h, w, c = FEATURES_SHAPE

x = Reshape((h * w, c), name="encoder_reshape")(image_features)
encoder_output = Dense(ATTENTION_DIM, activation="relu", name="encoder_dense")(x)
encoder = tf.keras.Model(inputs=image_input, outputs=encoder_output, name="encoder")
encoder.summary()


word_input = Input(shape=(MAX_CAPTION_LEN - 1,), name="words")
embed_layer = Embedding(VOCAB_SIZE, ATTENTION_DIM, name="word_embedding")
embed_x = embed_layer(word_input)  # (batch, T, ATTENTION_DIM)

decoder_gru = GRU(
    ATTENTION_DIM,
    return_sequences=True,
    return_state=True,
    name="decoder_gru",
)
gru_output, gru_state = decoder_gru(embed_x)

decoder_attention = Attention(name="decoder_attention")
context_vector = decoder_attention([gru_output, encoder_output])

addition = Add(name="decoder_add")([gru_output, context_vector])

layer_norm = LayerNormalization(axis=-1, name="decoder_layernorm")
layer_norm_out = layer_norm(addition)

decoder_output_dense = Dense(VOCAB_SIZE, name="decoder_output_dense")
decoder_output = decoder_output_dense(layer_norm_out)

decoder = tf.keras.Model(
    inputs=[word_input, encoder_output], outputs=decoder_output, name="decoder"
)
decoder.summary()

# Full training model: image + words -> logits
image_caption_train_model = tf.keras.Model(
    inputs=[image_input, word_input], outputs=decoder_output, name="image_caption_model"
)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)

def loss_function(real, pred):
    loss_ = loss_object(real, pred)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

image_caption_train_model.compile(
    optimizer="adam",
    loss=loss_function,
)

# Optional: run eagerly for easier debugging
image_caption_train_model.run_eagerly = False


'''history = image_caption_train_model.fit(
    batched_ds,
    epochs=1,     # increase as needed
)'''


inf_word_input = Input(shape=(1,), name="inf_word_input")
inf_gru_state_in = Input(shape=(ATTENTION_DIM,), name="inf_gru_state")
inf_encoder_output = Input(shape=(h * w, ATTENTION_DIM), name="inf_encoder_output")

inf_embed = embed_layer(inf_word_input)
inf_gru_output, inf_gru_state = decoder_gru(
    inf_embed, initial_state=inf_gru_state_in
)
inf_context = decoder_attention([inf_gru_output, inf_encoder_output])
inf_add = Add()([inf_gru_output, inf_context])
inf_ln = layer_norm(inf_add)
inf_logits = decoder_output_dense(inf_ln)

decoder_pred_model = tf.keras.Model(
    inputs=[inf_word_input, inf_gru_state_in, inf_encoder_output],
    outputs=[inf_logits, inf_gru_state],
    name="decoder_pred_model",
)

MINIMUM_SENTENCE_LENGTH = 5


def predict_caption(filename):
    gru_state = tf.zeros((1, ATTENTION_DIM))

    img = tf.image.decode_jpeg(tf.io.read_file(filename), channels=IMG_CHANNELS)
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0

    features = encoder(tf.expand_dims(img, axis=0))  # (1, h*w, ATTENTION_DIM)
    dec_input = tf.constant([[word_to_index("<start>")]], dtype=tf.int64)
    result = []

    for i in range(MAX_CAPTION_LEN - 1):
        logits, gru_state = decoder_pred_model(
            [dec_input, gru_state, features]
        )  # (1, 1, VOCAB_SIZE)

        logits = logits[0, 0]  # (VOCAB_SIZE,)
        top_probs, top_idxs = tf.math.top_k(logits, k=10, sorted=False)
        chosen_id = tf.random.categorical(tf.expand_dims(top_probs, 0), 1)[0, 0]
        predicted_id = top_idxs[chosen_id].numpy()

        word = tokenizer.get_vocabulary()[predicted_id]
        result.append(word)

        if predicted_id == word_to_index("<end>") and len(result) >= MINIMUM_SENTENCE_LENGTH:
            break

        dec_input = tf.constant([[predicted_id]], dtype=tf.int64)

    return img, result

