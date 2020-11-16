import math
from time import time
import numpy as np
import torch
from torch import nn
import torchvision.datasets as datasets


def run(NUM_CNN_LAYERS, CNN_FILTER_SIZE, OPTIMIZER):
    #
    # CONFIG
    #
    print("NUM_CNN_LAYERS =", NUM_CNN_LAYERS)
    print("CNN_FILTER_SIZE =", CNN_FILTER_SIZE)
    print("OPTIMIZER =", OPTIMIZER)

    start_time = time()

    #
    # Load MNIST Dataset
    #
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    # max_elements = 1000
    max_elements = 1000000  # i.e. use all of them.

    X_train = mnist_trainset.data[:max_elements]
    y_train = mnist_trainset.targets[:max_elements]
    X_test = mnist_testset.data[:max_elements]
    y_test = mnist_testset.targets[:max_elements]

    X_train = torch.reshape(X_train, (len(X_train), 1, 28, 28)).float()
    X_test = torch.reshape(X_test, (len(X_test), 1, 28, 28)).float()

    #
    # Create Model
    #
    w = {
        "0,0": 28 * 28,
        "1,3": 169,
        "1,5": 144,
        "1,9": 100,
        "2,3": 144,
        "2,5": 100,
        "2,9": 36,
    }[f"{NUM_CNN_LAYERS},{CNN_FILTER_SIZE}"]

    layers = []

    for i in range(NUM_CNN_LAYERS):
        layers.append(nn.Conv2d(1, 1, CNN_FILTER_SIZE))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(1))

    if NUM_CNN_LAYERS > 0:
        layers.append(nn.MaxPool2d(2))

    layers = layers + [
        nn.ReLU(),
        nn.Flatten(),

        nn.Linear(w, 10),
        nn.ReLU(),
    ]

    model = nn.Sequential(*layers)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) if OPTIMIZER == "Adam" else torch.optim.SGD(
        model.parameters(), lr=0.01)

    #
    # Train and evaluate model
    #
    l_loss = []
    l_acc = []
    l_val_loss = []
    l_val_acc = []

    batch_size = 2048
    for i in range(10):
        print("Iteration:", i)
        for j in range(0, len(X_train), batch_size):
            jS = j
            jE = min(len(X_train), j + batch_size)
            X_batch = X_train[jS:jE, :]
            y_batch = y_train[jS:jE]

            # clear previous gradient calculations
            optimizer.zero_grad()

            # calculate f based on current value
            y_pred = model(X_batch)
            y_true = y_batch
            z = loss(y_pred, y_true)

            # calculate gradient of f w.r.t. w
            z.backward()

            # use the gradient to change w
            optimizer.step()

            #
            # Calculate Training Error
            #
            # calculate f based on current value
        y_pred = model(X_train)
        y_true = y_train
        z = loss(y_pred, y_true)
        accuracy = sum((y_pred.argmax(axis=1) - y_true).abs() == 0) / float(len(y_train))
        l_loss.append(float(z))
        l_acc.append(float(accuracy))
        print("Training:  ", "loss:", float(z), "accuracy:", float(accuracy))

        #
        # Calculate Testing Error
        #
        y_pred = model(X_test)
        y_true = y_test
        z = loss(y_pred, y_true)
        accuracy = sum((y_pred.argmax(axis=1) - y_true).abs() == 0) / float(len(y_test))
        l_val_loss.append(float(z))
        l_val_acc.append(float(accuracy))
        print("Validation:", "loss:", float(z), "accuracy:", float(accuracy))

    #
    # Output Results for plotting
    #
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print("NUM_CNN_LAYERS =", NUM_CNN_LAYERS)
    print("CNN_FILTER_SIZE =", CNN_FILTER_SIZE)
    print("OPTIMIZER =", OPTIMIZER)
    print("time_taken =", time() - start_time)
    print("loss =", l_loss)
    print("validation_loss =", l_val_loss)
    print("acc =", l_acc)
    print("validation_acc =", l_val_acc)


if __name__ == "__main__":
    import sys

    NUM_CNN_LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    CNN_FILTER_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    OPTIMIZER = sys.argv[3] if len(sys.argv) > 3 else "Adam"
    if OPTIMIZER not in ["Adam", "SGD"]:
        raise Exception("Optimizer not allowed.")

    run(NUM_CNN_LAYERS, CNN_FILTER_SIZE, OPTIMIZER)
