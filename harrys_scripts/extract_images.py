import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


images = np.load("../mri_data/images.npy")
for i in range(images.shape[2]):
    im = images[:, :, i]
    im = (255.0 / im.max() * (im - im.min())).astype(np.uint8)
    im = Image.fromarray(im).convert('RGB')
    im.save(f"../mri_data/pngs/mri_{i}.png")