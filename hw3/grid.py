from _csv import writer
import sys
sys.path.append("../")

from hw3.solution import run

with open("grid.csv", "w+") as f:
    csv_writer = writer(f)

    # write header
    csv_writer.writerow(["n_epochs", "L", "C", "batch_size", "lr", "training_error"])

    for L in [10e6,10e8,10e10]:
        for C in [1e2, 1e3, 1e4]:
            print(L, C)
            training_error = run(
                n_epochs = 100,
                L = L,
                C = C,
                lr = 1e-6,
                batch_size = 2048,
            )
            csv_writer.writerow([1, L, C, 2048, 1e-6, training_error])

