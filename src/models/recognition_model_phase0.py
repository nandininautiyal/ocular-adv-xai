"""
src/models/recognition_model_phase0.py

Phase 0 smoke-test model: our own first re-implementation of Xception + ArcFace.
Kept only as a record of the Phase 0 plumbing check (scripts/00_sanity_check.py).
It differs from the paper's model (global average pooling instead of Flatten, no
partial freezing), so do NOT use it for experiments. Use recognition_model.py.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, Input  # type: ignore[reportMissingModuleSource]


class ArcFace(layers.Layer):
    """Additive Angular Margin (ArcFace) head.
    Expects [embeddings, one_hot_labels] as input during training.
    At inference, pass a zero one-hot vector to disable the margin term.
    """
    def __init__(self, num_classes, margin=0.5, scale=64.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

    def build(self, input_shape):
        embedding_dim = input_shape[0][-1]
        self.W = self.add_weight(
            name="arcface_weights",
            shape=(embedding_dim, self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        embeddings, one_hot_labels = inputs
        embeddings = tf.math.l2_normalize(embeddings, axis=1)
        w = tf.math.l2_normalize(self.W, axis=0)
        cosine = tf.matmul(embeddings, w)
        theta = tf.acos(tf.clip_by_value(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = tf.cos(theta + self.margin)
        one_hot = tf.cast(one_hot_labels, tf.float32)
        logits = one_hot * target_logits + (1.0 - one_hot) * cosine
        return logits * self.scale


def build_recognition_model(num_classes, input_shape=(256, 256, 3)):
    """Returns (training_model, embedding_model).
    training_model: [image, one_hot_label] -> class probabilities (use for fit()).
    embedding_model: image -> 1024-d embedding.
    """
    base = tf.keras.applications.Xception(
        include_top=False, weights="imagenet",
        input_shape=input_shape, pooling="avg",
    )
    image_input = Input(shape=input_shape, name="image")
    label_input = Input(shape=(num_classes,), name="label_onehot")

    x = base(image_input)
    embedding = layers.Dense(1024, activation="relu", name="embedding")(x)
    x = layers.Dropout(0.5)(embedding)
    logits = ArcFace(num_classes, name="arcface")([x, label_input])
    probs = layers.Softmax()(logits)

    training_model = Model(inputs=[image_input, label_input], outputs=probs, name="recognition_train")
    embedding_model = Model(inputs=image_input, outputs=embedding, name="recognition_embed")
    return training_model, embedding_model


def predict_labels(training_model, images, num_classes):
    """Inference helper: dummy zero one-hot disables the ArcFace margin,
    so this returns plain cosine-similarity-based class probabilities."""
    dummy_labels = tf.zeros((images.shape[0], num_classes))
    probs = training_model.predict([images, dummy_labels], verbose=0)
    return probs.argmax(axis=1)