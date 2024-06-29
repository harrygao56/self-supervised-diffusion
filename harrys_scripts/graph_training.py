import numpy as np
import matplotlib.pyplot as plt

arr = np.loadtxt("../logs/run1/progress.csv", delimiter=",", dtype=str)
losses = arr[1:, 1].float()
x = np.arange(len(losses))

print(x)
print(losses.shape)

plt.plot(x, losses)
plt.show()