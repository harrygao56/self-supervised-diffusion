import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


directory = "../mri_data/pngs"
images = np.zeros([len(os.listdir(directory)), 128, 128, 3])
for i, filename in enumerate(os.listdir(directory)):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        print(str(i) + " " + filename)
        img = Image.open(f)
        img = img.resize((128, 128))
        img = np.asarray(img)
        images[i] = img

np.savez("../mri_data/images_rgb.npz", images)