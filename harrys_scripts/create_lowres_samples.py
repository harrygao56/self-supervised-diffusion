# Creates a numpy array of 

from fastmri_brain import ALL_IDX_LIST, FastMRIBrain, uniformly_cartesian_mask, fmult, ftran
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

fastmri = FastMRIBrain(
    ALL_IDX_LIST, 1
)

num_samples = 10
resize_resolution = 256
sample_array = np.zeros((num_samples, resize_resolution, resize_resolution, 1))

mask = uniformly_cartesian_mask(fastmri[0]['x'].shape, 4).astype(int)

for i in range(num_samples):
    # Compute x_hat
    x = abs(fastmri[i]['x'])
    smps = fastmri[i]['smps']
    y = fmult(torch.unsqueeze(torch.from_numpy(x), 0), torch.unsqueeze(torch.from_numpy(smps), 0), torch.unsqueeze(torch.from_numpy(mask), 0))
    x_hat = ftran(y, torch.unsqueeze(torch.from_numpy(smps), 0), torch.unsqueeze(torch.from_numpy(mask), 0))
    x_hat = abs(x_hat.cpu().detach().numpy())[0,:,:]

    # Crop x_hat
    x_hat = x_hat[(x_hat.shape[0]-x_hat.shape[1])//2:(x_hat.shape[0]-x_hat.shape[1])//2+x_hat.shape[1],:]

    # Convert pad array to add 2 layers
    # x_hat = np.expand_dims(x_hat, axis=0)
    # x_hat = np.pad(x_hat, ((0, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)

    # Resize to dimension
    x_hat = F.interpolate(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(x_hat), 0), 0).float(), (resize_resolution, resize_resolution), mode="area").cpu().detach().numpy()[0,:,:,:]
    x_hat = np.transpose(x_hat, [1, 2, 0])

    print(x_hat.shape)
    
    sample_array[i] = x_hat

np.savez("harrys_scripts/fastmri_lowres_samples_10x256x256x1", sample_array)