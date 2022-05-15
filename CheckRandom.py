import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("NormalResults.txt")

print(f"Mean: {data.mean()}, Std: {data.std()}")

plt.ion()
plt.hist(data.flatten(), bins=100)