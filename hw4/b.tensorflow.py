from time import time
from tensorflow import keras
from tensorflow.keras import layers

def run(NUM_CNN_LAYERS, CNN_FILTER_SIZE):
    #
    # CONFIG
    #
    print("NUM_CNN_LAYERS =", NUM_CNN_LAYERS)
    print("CNN_FILTER_SIZE =", CNN_FILTER_SIZE)

    start_time = time()

    #
    # Build Model
    #
    inputs = keras.Input(shape=(28,28,1))
    x = inputs

    for i in range(NUM_CNN_LAYERS):
        if i == 0:
            x = layers.Conv2D(CNN_FILTER_SIZE, CNN_FILTER_SIZE, input_shape=(28,28,1))(x)
        else:
            x = layers.Conv2D(CNN_FILTER_SIZE, CNN_FILTER_SIZE)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)

    if NUM_CNN_LAYERS > 0:
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)

    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        metrics=["accuracy"],
    )

    #
    # Setup Dataset
    #
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype(dtype='float32')
    x_test = x_test.astype(dtype='float32')
    y_train = y_train.astype(dtype='float32')
    y_test = y_test.astype(dtype='float32')

    x_train = x_train.reshape(len(x_train), 28, 28, 1)
    x_test = x_test.reshape(len(x_test), 28, 28, 1)

    # max_elements = 100
    max_elements = 1000000  # a.k.a. use all of them.

    x_train = x_train[:max_elements]
    y_train = y_train[:max_elements]
    x_test = x_test[:max_elements]
    y_test = y_test[:max_elements]

    #
    # Train and Evaluate
    #
    history = model.fit(x_train, y_train, batch_size=2048, epochs=10, validation_data=(x_test, y_test))
    print("history", history.history)

    results = model.evaluate(x_test, y_test, verbose=2)
    print("Validation loss:", results[0])
    print("Validation accuracy:", results[1])

    #
    # Output Results for plotting
    #
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print("NUM_CNN_LAYERS =", NUM_CNN_LAYERS)
    print("CNN_FILTER_SIZE =", CNN_FILTER_SIZE)
    print("time_taken =", time() - start_time)
    print("loss =", history.history['loss'])
    print("validation_loss =", history.history['val_loss'])
    print("acc =", history.history['accuracy'])
    print("validation_acc =", history.history['val_accuracy'])

if __name__ == "__main__":
  import sys

  NUM_CNN_LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 0
  CNN_FILTER_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 0

  run(NUM_CNN_LAYERS, CNN_FILTER_SIZE)
