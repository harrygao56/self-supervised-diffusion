import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

images = np.load('samples/fastmri-run5-260000-new_timestep/samples_1x256x256x3.npz')
# images = np.load('harrys_scripts/fastmri_lowres_samples_10x256x256x1.npz')
print(images['arr_0'].shape)
for ind in range(len(images['arr_0'])):
    x = images['arr_0'][ind, :, :, 0]

    plt.imshow(x, cmap='gray')
    plt.show()
    
    plt.savefig(f"samples/fastmri-run5-260000-new_timestep/image{ind}.png")
    # plt.savefig(f"harrys_scripts/lowres_samples/image{ind}.png")