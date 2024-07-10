import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from guided_diffusion.fastmri_dataloader import (FastBrainMRI, uniformly_cartesian_mask, fmult, ftran, get_weighted_mask)
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def load_data(
    *,
    data_dir,
    batch_size,
    dataset_type="",
    indi="",
    split="",
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if indi == True:
        print("creating indi dataset")
        dataset = InDIDataset(
            split
        )
    elif dataset_type == "supervised":
        print("creating supervised dataset")
        dataset = FastMRIDataset(
            split
        )
    elif dataset_type == "ambient":
        print("creating ambient dataset")
        dataset = AmbientDataset(
            split
        )
    elif dataset_type == "fullrank":
        print("creating fullrank dataset")
        dataset = FullRankDataset(
            split
        )
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "gz"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class FastMRIDataset(FastBrainMRI):
    # Dataset for conditional diffusion
    def __init__(
            self,
            split,
            acceleration_rate=4,
            noise_sigma=0.01,
            is_return_y_smps_hat=True,
        ):
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, is_return_y_smps_hat=is_return_y_smps_hat)

    def __getitem__(self, item):
        x, x_hat, _, _, _, _, _, _ = super().__getitem__(item)

        out_dict = {
            "AtAx": x_hat
        }
        return x, out_dict


class InDIDataset(FastBrainMRI):
    def __init__(
            self,
            split,
            acceleration_rate=4,
            noise_sigma=0.01,
            is_return_y_smps_hat=True,
        ):
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, is_return_y_smps_hat=is_return_y_smps_hat)

    def __getitem__(self, item):
        x, x_hat, _, _, _, _, _, _ = super().__getitem__(item)
        x = torch.cat([x.real, x.imag], axis=0)
        x_hat = torch.cat([x_hat.real, x_hat.imag], axis=0)
        out_dict = {
            "AtAx": x_hat
        }
        return x, out_dict
    

class AmbientDataset(FastBrainMRI):
    def __init__(
            self,
            split,
            acceleration_rate=4,
            noise_sigma=0.01,
            is_return_y_smps_hat=True,
        ):
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, is_return_y_smps_hat=is_return_y_smps_hat)

    def __getitem__(self, item):
        x, x_hat, smps, smps_hat, y, mask, _, _ = super().__getitem__(item)

        mask_noisier = uniformly_cartesian_mask(x.shape, 8).astype(int)
        mask_noisier = torch.from_numpy(mask_noisier)

        out_dict = {
            "smps": smps, 
            "A": mask,
            "A_hat": mask_noisier,
            "AtAx": x_hat,
        }
        return x, out_dict


class FullRankDataset(FastBrainMRI):
    def __init__(
            self,
            split,
            acceleration_rate=0,
            noise_sigma=0.01,
            is_return_y_smps_hat=True,
            preload=True,
        ):
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, is_return_y_smps_hat=is_return_y_smps_hat)

            # If preload is set, we preload the masks to make sure we apply the same masks each time
            self.preload = preload
            if preload:
                self.preloaded_masks = []
                mask_shape = super().__getitem__(0)[0].shape
                for i in range(len(self)):
                    self.preloaded_masks.append(uniformly_cartesian_mask(mask_shape, 4, randomly_return=True, get_two=True))

    def __getitem__(self, item):
        x, _, smps, _, _, _, _, _ = super().__getitem__(item)

        # Generate a cartesian mask for the groundtruth
        if self.preload:
            A, A_hat = self.preloaded_masks[item]
        else:
            A, A_hat = uniformly_cartesian_mask(x.shape, 4, randomly_return=True, get_two=True)
        
        A = torch.from_numpy(A.astype(int))
        A_hat = torch.from_numpy(A_hat.astype(int))

        # Apply mask in k-space, then use ftran to convert back to image space
        Ax = fmult(torch.unsqueeze(torch.from_numpy(x), 0), smps, torch.unsqueeze(A, 0))
        AtAx = ftran(Ax, smps, torch.unsqueeze(A, 0))

        W = torch.from_numpy(get_weighted_mask(x.shape, 4).astype(int))
        out_dict = {
            "smps": smps, 
            "A": A,
            "A_hat": A_hat,
            "AtAx": AtAx,
            "W": W,
        }
        return x, out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def main():
    dataset = FullRankDataset(
            256, "train"
        )

    testset = FullRankDataset(
            256, "test"
        )

    print(len(dataset))
    print(len(testset))

    # print(args['x'].shape)
    # print(args['Ax'].shape)
    # print(args['mask'].shape)
    # print(im.shape)

    # plt.imshow(im[0,:,:], cmap='gray')
    # plt.show()
    # plt.savefig("x-hat")
    # plt.imshow(args["Ax"][0,:,:], cmap='gray')
    # plt.show()
    # plt.savefig("x-hat-masked")
    # plt.imshow(args["x"], cmap='gray')
    # plt.show()
    # plt.savefig("x")


if __name__ == "__main__":
    main()