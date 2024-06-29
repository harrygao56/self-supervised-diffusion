from fastmri_brain import ALL_IDX_LIST, FastMRIBrain, uniformly_cartesian_mask, fmult, ftran
import numpy as np
import matplotlib.pyplot as plt
import torch

fastmri = FastMRIBrain(
    ALL_IDX_LIST, 1
)

print(len(fastmri))

for img in fastmri:
    print(img['x'].shape)

# curr_ind = 2

# test = abs(fastmri[curr_ind]['x'])
# print(test.shape)

# mask = uniformly_cartesian_mask(test.shape, 4).astype(int)

# y = fmult(torch.unsqueeze(torch.from_numpy(test), 0), torch.unsqueeze(torch.from_numpy(fastmri[curr_ind]['smps']), 0), torch.unsqueeze(torch.from_numpy(mask), 0))
# # y is in k-space, need to use ftran to convert it to image space
# x_hat = ftran(y, torch.unsqueeze(torch.from_numpy(fastmri[curr_ind]['smps']), 0), torch.unsqueeze(torch.from_numpy(mask), 0))
# x_hat = abs(x_hat.cpu().detach().numpy())[0,:,:]

# print(x_hat.shape)

# # Crop images to square
# test = test[(test.shape[0]-test.shape[1])//2:(test.shape[0]-test.shape[1])//2+test.shape[1],:]
# x_hat = x_hat[(x_hat.shape[0]-x_hat.shape[1])//2:(x_hat.shape[0]-x_hat.shape[1])//2+x_hat.shape[1],:]

# print(x_hat.dtype)

# print(x_hat.shape)
# print(test.shape)

# plt.imshow(test, cmap="viridis")
# plt.show()
# plt.savefig("image.png")

# plt.imshow(x_hat, cmap="viridis")
# plt.show()
# plt.savefig("x_hat.png")