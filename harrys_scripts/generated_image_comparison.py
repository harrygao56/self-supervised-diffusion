# Script to check if diffusion model is generating novel images or
# recreating existing ones in the training set

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageChops
import math


def rmsdiff(im1, im2):
    "Calculate the root-mean-square difference between two images"
    diff = ImageChops.difference(im1, im2)
    h = diff.histogram()
    sq = (value*((idx%256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(im1.size[0] * im1.size[1]))
    return rms

generated_images = np.load('../samples/log1/280000/samples_100x128x128x3.npz')
generated_images_processed = []
for ind in range(len(generated_images['arr_0'])):
    im = generated_images['arr_0'][ind, :, :, :].astype(np.uint8)
    im = Image.fromarray(im)
    generated_images_processed.append(im)

training_images = np.load("../mri_data/images.npy")
training_images_processed =[]
for i in range(training_images.shape[2]):
    im = training_images[:, :, i]
    im = (255.0 / im.max() * (im - im.min())).astype(np.uint8)
    im = Image.fromarray(im).convert('RGB')
    training_images_processed.append(im)

lowest = float('inf')
train_ind = 0
gen_ind = 0

for i in range(len(training_images_processed)):
    for j in range(len(generated_images_processed)):
        rms = rmsdiff(training_images_processed[i], generated_images_processed[j])
        if rms < lowest:
            lowest = rms
            train_ind = i
            gen_ind = j


print(lowest)
print(rmsdiff(training_images_processed[train_ind], generated_images_processed[gen_ind]))
training_images_processed[train_ind].save("closest_train.jpg")
generated_images_processed[gen_ind].save("generated.jpg")