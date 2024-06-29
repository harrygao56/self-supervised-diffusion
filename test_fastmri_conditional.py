from guided_diffusion.image_datasets import FastMRIDataset
import numpy as np
import matplotlib.pyplot as plt
import torch

fastmri = FastMRIDataset(396)

print(len(fastmri))
curr_ind = 0

img, data_dict = fastmri[curr_ind]

plt.imshow(img, cmap="viridis")
plt.show()
plt.savefig("img.png")

plt.imshow(data_dict["low_res"], cmap="viridis")
plt.show()
plt.savefig("low_res.png")