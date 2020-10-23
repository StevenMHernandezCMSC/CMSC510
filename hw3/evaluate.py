import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("grid.csv")
df.training_error = df.training_error.apply(json.loads)

L_s = df.L.unique()
C_s = df.C.unique()

grid = np.zeros((len(L_s), len(C_s)))

for i, L in enumerate(L_s):
    for j, C in enumerate(C_s):
        val = df[df.C == C][df.L == L].training_error.apply(lambda x: x[-1])
        if len(val):
            grid[i,j] = val

fig, ax = plt.subplots()
ax.imshow(grid)

for i, L in enumerate(L_s):
    for j, C in enumerate(C_s):
        text = ax.text(j, i, grid[i, j], ha="center", va="center", color="w")

plt.xticks(range(len(C_s)), C_s)
plt.xlabel("C")
plt.yticks(range(len(L_s)), L_s)
plt.ylabel("L")
plt.show()