import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def run():
    inputs = keras.Input(shape=(28,28,1))
    x = layers.Conv2D(3, 3, input_shape=(28,28,1))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Conv2D(3, 3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

    keras.utils.plot_model(model, "model.png", show_shapes=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype(dtype='float32')
    x_test = x_test.astype(dtype='float32')
    y_train = y_train.astype(dtype='float32')
    y_test = y_test.astype(dtype='float32')

    x_train = x_train.reshape(len(x_train), 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(len(x_test), 28, 28, 1).astype("float32") / 255

    # max_elements = 1000
    max_elements = 1000000  # a.k.a. use all of them.

    x_train = x_train[:max_elements]
    y_train = y_train[:max_elements]
    x_test = x_test[:max_elements]
    y_test = y_test[:max_elements]

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.2)

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

if __name__ == "__main__":
    run()
