import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("grid.csv")
# df.training_error = df.training_error.apply(json.loads)

# L_s = df.L.unique()
# C_s = df.C.unique()

# grid = np.zeros((len(L_s), len(C_s)))

# for i, L in enumerate(L_s):
#     for j, C in enumerate(C_s):
#         val = df[df.C == C][df.L == L].training_error.apply(lambda x: x[-1])
#         if len(val):
#             grid[i,j] = val

# fig, ax = plt.subplots()
# ax.imshow(grid)

# for i, L in enumerate(L_s):
#     for j, C in enumerate(C_s):
#         text = ax.text(j, i, grid[i, j], ha="center", va="center", color="w")

# plt.xticks(range(len(C_s)), C_s)
# plt.xlabel("C")
# plt.yticks(range(len(L_s)), L_s)
# plt.ylabel("L")
# plt.show()



# legends = []
# for i, L in enumerate(L_s):
#     for j, C in enumerate(C_s):
#         val = df[df.C == C][df.L == L].training_error
#         plt.plot(1 - (np.array(list(val)[0]) / 1901))
#         L_str = f"1e{(i*0.5)+5}"
#         legends.append(f"C={C}, L={L_str}")
# plt.legend(legends)
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.show()


df = pd.read_csv("final_accuracies.csv")
legends = []
for i,row in df.iterrows():
    plt.plot(sorted([1-(float(x)/1901) for x in row.accuracy.strip('][').split(', ')]))
    legends.append(f"L1 Penalty: {row.use_l1_penalty}, Proximal: {row.use_proximal_soft_thresholding}")
plt.legend(legends)
plt.xlabel("epochs")
plt.ylabel("Accuracy (%)")
plt.show()

# df = pd.read_csv("final_weights.csv")
# legends = []
# for i,row in df.iterrows():
#     plt.plot(sorted([float(x) for x in row.w.strip('][').split(', ')]))
#     legends.append(f"L1 Penalty: {row.use_l1_penalty}, Proximal: {row.use_proximal_soft_thresholding}")
# plt.legend(legends)
# plt.xlabel("i")
# plt.ylabel("w CDF")
# plt.show()
# df = pd.read_csv("final_weights.csv")