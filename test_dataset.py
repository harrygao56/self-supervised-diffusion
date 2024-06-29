from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from guided_diffusion.image_datasets import CustomImageDataset, _list_image_files_recursively
import matplotlib.pyplot as plt


all_files = _list_image_files_recursively("../mri_dec_data/imagesTr")

dataset = CustomImageDataset(
    resolution=128,
    image_paths=all_files,
    classes=None,
    shard=MPI.COMM_WORLD.Get_rank(),
    num_shards=MPI.COMM_WORLD.Get_size(),
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
data_iter = iter(dataloader)
images, labels = next(data_iter)
for image in images:
    image = image.cpu().numpy()
    print(image.shape)
    plt.imshow(image)
    plt.show()