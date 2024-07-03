import h5py
import os
import numpy as np
import tqdm
import torch
import tifffile
from torch.utils.data import Dataset
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

ROOT_PATH_DEFAULT = '/project/cigserver5/export1/gan.weijie/dataset/fastmri_brain/'
DATASHEET_DEFAULT = '20240506_default.csv'


def ALL_IDX_LIST(datasheet):
    return list(datasheet['INDEX'])


def INDEX2_helper(idx, key_, datasheet):
    file_id_df = datasheet[key_][datasheet['INDEX'] == idx]

    assert len(file_id_df) == 1

    return file_id_df.item()


INDEX2FILE = lambda idx, datasheet: INDEX2_helper(idx, 'FILE', datasheet)


def INDEX2DROP(idx, datasheet):
    ret = INDEX2_helper(idx, 'DROP', datasheet)

    if ret in ['0', 'false', 'False', 0.0]:
        return False
    else:
        return True


def INDEX2SLICE_START(idx, datasheet):
    ret = INDEX2_helper(idx, 'SLICE_START', datasheet)

    if isinstance(ret, np.float64) and ret >= 0:
        return int(ret)
    else:
        return None


def INDEX2SLICE_END(idx, datasheet):
    ret = INDEX2_helper(idx, 'SLICE_END', datasheet)

    if isinstance(ret, np.float64) and ret >= 0:
        return int(ret)
    else:
        return None


def ftran(y, smps, mask):
    """
    compute adjoint of fast MRI, x = smps^H F^H mask^H x

    :param y: under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: zero-filled image
    """

    # mask^H
    y = y * mask.unsqueeze(1)

    # F^H
    y = torch.fft.ifftshift(y, [-2, -1])
    x = torch.fft.ifft2(y, norm='ortho')
    x = torch.fft.fftshift(x, [-2, -1])

    # smps^H
    x = x * torch.conj(smps)
    x = x.sum(1)

    return x


def fmult(x, smps, mask):
    """
    compute forward of fast MRI, y = mask F smps x

    :param x: groundtruth or estimated image, shape: batch, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: undersampled measurement
    """

    # smps
    x = x.unsqueeze(1)
    y = x * smps

    # F
    y = torch.fft.ifftshift(y, [-2, -1])
    y = torch.fft.fft2(y, norm='ortho')
    y = torch.fft.fftshift(y, [-2, -1])

    # mask
    mask = mask.unsqueeze(1)
    y = y * mask

    return y


def uniformly_cartesian_mask(img_size, acceleration_rate, acs_percentage: float = 0.2, randomly_return: bool = False, get_two: bool = False):

    ny = img_size[-1]

    ACS_START_INDEX = (ny // 2) - (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)
    ACS_END_INDEX = (ny // 2) + (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)

    if ny % 2 == 0:
        ACS_END_INDEX -= 1

    mask = np.zeros(shape=(acceleration_rate,) + img_size, dtype=np.float32)
    mask[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1

    for i in range(ny):
        for j in range(acceleration_rate):
            if i % acceleration_rate == j:
                mask[j, ..., i] = 1

    if randomly_return:
        if get_two:
            np.random.seed(0)
            n1, n2 = np.random.choice(np.arange(0, acceleration_rate), size=2, replace=False)
            return mask[n1], mask[n2]
        else:
            mask = mask[np.random.randint(0, acceleration_rate)]
    else:
        mask = mask[0]

    return mask


_mask_fn = {
    'uniformly_cartesian_randomly_False': lambda img_size, acceleration_rate: uniformly_cartesian_mask(img_size=img_size, acceleration_rate=acceleration_rate, randomly_return=False),

    'uniformly_cartesian_randomly_True': lambda img_size, acceleration_rate: uniformly_cartesian_mask(img_size=img_size, acceleration_rate=acceleration_rate, randomly_return=True),
}

def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def np_normalize_to_uint8(x):
    x -= np.amin(x)
    x /= np.amax(x)

    x = x * 255
    x = x.astype(np.uint8)

    return x


def torch_complex_normalize(x):
    x_angle = torch.angle(x)
    x_abs = torch.abs(x)

    x_abs -= torch.min(x_abs)
    x_abs /= torch.max(x_abs)

    x = x_abs * np.exp(1j * x_angle)

    return x


def addwgn(x: torch.Tensor, input_snr):
    noiseNorm = torch.norm(x.flatten()) * 10 ** (-input_snr / 20)

    noise = torch.randn(x.size()).to(x.device)

    noise = noise / torch.norm(noise.flatten()) * noiseNorm

    y = x + noise
    return y


def load_real_dataset_handle(
        idx,
        datasheet,
        root_path=ROOT_PATH_DEFAULT,
        smps_hat_method: str = 'eps',
        acceleration_rate: int = 2,
        mask_pattern: str = 'uniformly_cartesian_randomly_True',
        noise_sigma= 5 / 255,
        is_qc: bool = False,
):

    assert noise_sigma < 1

    ret_dict = {}

    if not smps_hat_method == 'eps':  
        raise NotImplementedError('smps_hat_method can only be eps now, but found %s' % smps_hat_method)
    
    raw_y_h5 = os.path.join(root_path, INDEX2FILE(idx, datasheet) + '.h5')

    root_path = os.path.join(root_path, 'generated_dataset_h5')
    check_and_mkdir(root_path)

    """
    Ground-truth
    """
    groundtruth_path = os.path.join(root_path, 'groundtruth')
    check_and_mkdir(groundtruth_path)
    
    groundtruth_h5 = os.path.join(groundtruth_path, INDEX2FILE(idx, datasheet).split('/')[-1] + '.h5')
    
    groundtruth_path_qc = os.path.join(root_path, 'groundtruth_qc')
    check_and_mkdir(groundtruth_path_qc)
    
    groundtruth_qc_name = os.path.join(groundtruth_path_qc, INDEX2FILE(idx, datasheet).split('/')[-1])
    
    if not os.path.exists(groundtruth_h5):
        
        print(f"Don't find groundtruth_h5 of {groundtruth_h5}. start generating it")

        with h5py.File(raw_y_h5, 'r') as f:
            y = f['kspace'][:]

            # Normalize the kspace to 0-1 region
            for i in range(y.shape[0]):
                y[i] /= np.amax(np.abs(y[i]))
        
        _, _, n_x, n_y = y.shape
        if n_x == 768 and n_y == 396:
            y = y[:, :, 64:-64, 38:-38]

        os.environ['CUPY_CACHE_DIR'] = '/tmp/cupy'
        os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba'

        from sigpy.mri.app import EspiritCalib
        from sigpy import Device
        import cupy

        num_slice = y.shape[0]
        iter_ = tqdm.tqdm(range(num_slice), desc='[%s] estimate smps' % INDEX2FILE(idx, datasheet))

        smps = np.zeros_like(y)
        for i in iter_:
            tmp = EspiritCalib(y[i], device=Device(0), show_pbar=False).run()
            tmp = cupy.asnumpy(tmp)
            smps[i] = tmp

        y = torch.from_numpy(y)
        smps = torch.from_numpy(smps)
        
        _, _, n_x, n_y = y.shape
        x = ftran(y, smps, mask=torch.ones(size=(n_x, n_y), dtype=torch.float32).unsqueeze(0))

        x = x[..., (n_x // 4): -(n_x // 4), :]
        smps = smps[..., (n_x // 4): -(n_x // 4), :]

        for i in range(x.shape[0]):
            x[i] = torch_complex_normalize(x[i])
        
        data_dict = {
            'x': x,
            'smps': smps
        }
        with h5py.File(groundtruth_h5, 'w') as f:
            for k in data_dict:
                f.create_dataset(name=k, data=data_dict[k])

        if is_qc:
            for k in data_dict:
                tifffile.imwrite(f'{groundtruth_qc_name}_{k}.tiff', abs(data_dict[k].numpy()))

    ret_dict.update({
        'groundtruth': groundtruth_h5
    })
    
    if acceleration_rate <= 1:
        return ret_dict

    """
    Mask and y
    """
    mask_path = os.path.join(root_path, "noise_sigma_%s_mask_pattern_%s_acceleration_rate_%d" % (
        noise_sigma, mask_pattern, acceleration_rate))
    check_and_mkdir(mask_path)

    mask_h5 = os.path.join(mask_path, INDEX2FILE(idx, datasheet).split('/')[-1] + '.h5')

    mask_path_qc = os.path.join(root_path, "noise_sigma_%s_mask_pattern_%s_acceleration_rate_%d_qc" % (
        noise_sigma, mask_pattern, acceleration_rate))
    check_and_mkdir(mask_path_qc)

    mask_qc_name = os.path.join(mask_path_qc, INDEX2FILE(idx, datasheet).split('/')[-1])

    if not os.path.exists(mask_h5):
        
        print(f"Don't find mask_h5 of {mask_h5}. start generating it")

        with h5py.File(groundtruth_h5, 'r') as f:
            n_z, n_x, n_y = f['x'].shape

        mask = []
        for i in range(n_z):
            mask.append(_mask_fn[mask_pattern]((n_x, n_y), acceleration_rate))
        mask = np.stack(mask, 0)

        mask = torch.from_numpy(mask)

        with h5py.File(groundtruth_h5, 'r') as f:
            x = torch.from_numpy(f['x'][:])
            smps = torch.from_numpy(f['smps'][:])

        y = fmult(x, smps, mask)
        if noise_sigma > 0:
            y = y + torch.randn_like(y) * noise_sigma

        data_dict = {
            'mask': mask,
            'y': y
        }
        with h5py.File(mask_h5, 'w') as f:
            for k in data_dict:
                f.create_dataset(name=k, data=data_dict[k])

        data_dict = {
            'mask': mask,
        }
        if is_qc:
            for k in data_dict:
                tifffile.imwrite(f'{mask_qc_name}_{k}.tiff', abs(data_dict[k].numpy()))

    ret_dict.update({
        'mask_y': mask_h5
    })

    """
    Estimated coil sensitivity map
    """
    smps_hat_path = os.path.join(mask_path, "smps_hat_method_%s" % smps_hat_method)
    check_and_mkdir(smps_hat_path)

    smps_hat_h5 = os.path.join(smps_hat_path, INDEX2FILE(idx, datasheet).split('/')[-1] + '.h5')

    smps_hat_path_qc = os.path.join(mask_path, "smps_hat_method_%s_qc" % smps_hat_method)
    check_and_mkdir(smps_hat_path_qc)

    smps_hat_qc_name = os.path.join(smps_hat_path_qc, INDEX2FILE(idx, datasheet).split('/')[-1])

    if not os.path.exists(smps_hat_h5):
        
        print(f"Don't find smps_hat_h5 of {smps_hat_h5}. start generating it")

        with h5py.File(mask_h5, 'r') as f:
            y = f['y'][:]
            mask = f['mask'][:]
    
        os.environ['CUPY_CACHE_DIR'] = '/tmp/cupy'
        os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba'
        from sigpy.mri.app import EspiritCalib
        from sigpy import Device
        import cupy
        
        num_slice = y.shape[0]
        iter_ = tqdm.tqdm(range(num_slice), desc='[%s] estimate smps_hat' % INDEX2FILE(idx, datasheet))

        if smps_hat_method == 'eps':
            smps_hat = np.zeros_like(y)
            for i in iter_:
                tmp = EspiritCalib(y[i], device=Device(0), show_pbar=False).run()
                tmp = cupy.asnumpy(tmp)
                smps_hat[i] = tmp
        else:
            raise NotImplementedError()

        data_dict = {
            'smps_hat': smps_hat
        }
        with h5py.File(smps_hat_h5, 'w') as f:
            for k in data_dict:
                f.create_dataset(name=k, data=data_dict[k])

        y = torch.from_numpy(y)
        mask = torch.from_numpy(mask)
        smps_hat = torch.from_numpy(smps_hat)

        x_hat = ftran(y, smps_hat, mask)
        data_dict = {
            'smps_hat': smps_hat,
            'x_hat': x_hat
        }

        if is_qc:
            for k in data_dict:
                tifffile.imwrite(f'{smps_hat_qc_name}_{k}.tiff', abs(data_dict[k].numpy()))

    ret_dict.update({
        'smps_hat': smps_hat_h5
    })

    return ret_dict


def get_subsets(lst, n, subset_idx_start, subset_idx_end):
    """
    Get subsets of every nth element of a list.

    Parameters:
        lst (list): The input list.
        n (int): The step size.
        subset_length (int): The length of each subset.

    Returns:
        list: A list of subsets.
    """
    ret = []
    for i in range(0, len(lst), n):
        ret += lst[i+subset_idx_start:i+subset_idx_end]
    return ret


def get_weighted_mask(img_size, acceleration_rate, acs_percentage: float = 0.2):

    ny = img_size[-1]

    ACS_START_INDEX = (ny // 2) - (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)
    ACS_END_INDEX = (ny // 2) + (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)

    if ny % 2 == 0:
        ACS_END_INDEX -= 1

    mask = np.zeros(shape=(acceleration_rate,) + img_size, dtype=np.float32)
    mask[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1

    for i in range(ny):
        for j in range(acceleration_rate):
            if i % acceleration_rate == j:
                mask[j, ..., i] = 1
    mask = 1 / np.mean(mask, axis=0)
    return mask


class FastBrainMRI(Dataset):
    
    idx_subset_dict = {
        'tra_large': get_subsets(list(range(1, 6405 + 1)), 12, 0, 10),
        'val_large': get_subsets(list(range(1, 6405 + 1)), 12, 10, 11),
        'tst_large': get_subsets(list(range(1, 6405 + 1)), 12, 11, 12),

        'tra_medium': get_subsets(get_subsets(list(range(1, 6405 + 1)), 12, 0, 10), 4, 0, 1),
        'val_medium': get_subsets(get_subsets(list(range(1, 6405 + 1)), 12, 10, 11), 4, 0, 1),
        'tst_medium': get_subsets(get_subsets(list(range(1, 6405 + 1)), 12, 11, 12), 4, 0, 1),

        'tra_small': get_subsets(get_subsets(list(range(1, 6405 + 1)), 12, 0, 10), 10, 0, 1),
        'val_small': get_subsets(get_subsets(list(range(1, 6405 + 1)), 12, 10, 11), 10, 0, 1),
        'tst_small': get_subsets(get_subsets(list(range(1, 6405 + 1)), 12, 11, 12), 10, 0, 1),
    }
 
    def __init__(
            self,
            idx_subset,
            acceleration_rate=4,
            datasheet=None,
            is_return_y_smps_hat: bool = False,
            noise_sigma=0.01,
            mask_pattern: str = 'uniformly_cartesian_randomly_True',
            smps_hat_method: str = 'eps',
            num_coil_subset=None,
            image_size_subset=None,
            modality_subset=None,
            is_verbose=True,
            is_pre_load=False
    ):

        if datasheet is None:
            datasheet = DATASHEET_DEFAULT
        
        datasheet = pd.read_csv(os.path.join(ROOT_PATH_DEFAULT, 'datasheet', datasheet))

        assert image_size_subset in [None, 320, 396]
        assert modality_subset in [None, 'T2', 'T1', 'FLAIR', 'T1POST', 'T1PRE']
        if num_coil_subset is not None:
            assert isinstance(num_coil_subset, int)

        idx_list = self.idx_subset_dict[idx_subset]

        self.acceleration_rate = acceleration_rate
        self.is_return_y_smps_hat = is_return_y_smps_hat

        self.index_maps = []
        if is_verbose:
            iter_ = tqdm.tqdm(idx_list, desc='retriving idx')
        else:
            iter_ = idx_list
        for idx in iter_:
            
            # filter subset of specific coil number
            if num_coil_subset is not None:
                if INDEX2_helper(idx=idx, datasheet=datasheet, key_='COIL') != num_coil_subset:
                    continue
            
            # drop abnormal image size
            if INDEX2_helper(idx=idx, datasheet=datasheet, key_='Y') not in [320, 396]:
                continue
            
            if INDEX2_helper(idx=idx, datasheet=datasheet, key_='Y') == 320 and INDEX2_helper(idx=idx, datasheet=datasheet, key_='X') != 640:
                continue

            if INDEX2_helper(idx=idx, datasheet=datasheet, key_='Y') == 396 and INDEX2_helper(idx=idx, datasheet=datasheet, key_='X') != 768:
                continue
            
            # filter subset of specific image size
            if image_size_subset is not None:
                if INDEX2_helper(idx=idx, datasheet=datasheet, key_='Y') != image_size_subset:
                    continue
            
            # filter subset of specific image modality
            if modality_subset is not None:
                if f"{modality_subset}_" not in INDEX2_helper(idx=idx, datasheet=datasheet, key_='FILE'):
                    continue

            ret = load_real_dataset_handle(
                idx=idx, 
                datasheet=datasheet,
                smps_hat_method = smps_hat_method,
                acceleration_rate = acceleration_rate,
                mask_pattern = mask_pattern,
                noise_sigma=noise_sigma,
                is_qc=False
            )

            with h5py.File(ret['groundtruth'], 'r') as f:
                num_slice = f['x'].shape[0]

            # if INDEX2SLICE_START(idx) is not None:
            #     slice_start = INDEX2SLICE_START(idx)
            # else:
            #     slice_start = 0

            # if INDEX2SLICE_END(idx) is not None:
            #     slice_end = INDEX2SLICE_END(idx)
            # else:
            #     slice_end = num_slice - 5

            slice_start = 0
            slice_end = num_slice - 5

            for s in range(slice_start, slice_end):
                self.index_maps.append([idx, ret, s])

        self.get_item_buffer = []
        if is_pre_load:
            for item in tqdm.tqdm(range(len(self)), desc="Pre-loading"):
                self.get_item_buffer.append(self.__getitem__helper(item))
        self.is_pre_load = is_pre_load

    def quality_control_GUI(self):
        
        def process_fn(image_sets):
            if self.is_return_y_smps_hat:
                x, x_hat, smps, smps_hat, y, mask, idx, s = image_sets

                return {
                    f'x_{idx}_{s}': abs(x.squeeze().numpy()),
                    'x_hat': abs(x_hat.squeeze().numpy()),
                    'mask': abs(mask.squeeze().numpy()),
                    'smps #1': abs(smps.squeeze()[0].numpy()),
                    'smps_hat #1': abs(smps_hat.squeeze()[0].numpy()),
                }
            
            else:
                x, idx, s = image_sets

                return {
                    f'x_{idx}_{s}': abs(x.squeeze().numpy())
                }
            
        ImageSetViewer(self, process_fn)

    def __len__(self):

        return len(self.index_maps)

    def __getitem__(self, item):
        if self.is_pre_load:
            return self.get_item_buffer[item]
        else:
            return self.__getitem__helper(item)

    def __getitem__helper(self, item):

        idx, ret, s = self.index_maps[item]

        with h5py.File(ret['groundtruth'], 'r', swmr=True) as f:
            x = torch.from_numpy(f['x'][s]).unsqueeze(0)

        if self.is_return_y_smps_hat:
            with h5py.File(ret['groundtruth'], 'r', swmr=True) as f:
                smps = torch.from_numpy(f['smps'][s]).unsqueeze(0)
        
            with h5py.File(ret['mask_y'], 'r', swmr=True) as f:
                mask = torch.from_numpy(f['mask'][s]).unsqueeze(0)
                y = torch.from_numpy(f['y'][s]).unsqueeze(0)

            with h5py.File(ret['smps_hat'], 'r', swmr=True) as f:
                smps_hat = torch.from_numpy(f['smps_hat'][s]).unsqueeze(0)

            x_hat = ftran(y, smps_hat, mask)

            return x, x_hat, smps, smps_hat, y, mask, idx, s

        else:

            return x, idx, s


class ImageSetViewer:
    def __init__(self, image_sets: Dataset, process_fn):
        self.image_sets = image_sets
        self.process_fn = process_fn

        self.current_set_index = 0

        # Set up the figure and axes
        self.fig, self.axes = plt.subplots(1, len(self.process_fn(self.image_sets[0])), figsize=(15, 5))
        plt.subplots_adjust(bottom=0.2)

        if isinstance(self.axes, np.ndarray):
            self.imshow_return = []
            for ax in self.axes:
                self.imshow_return.append(ax.imshow(np.random.randn(256, 256), cmap='gray'))
                ax.set_title("")
                ax.axis('off')
        else:
            self.imshow_return = self.axes.imshow(np.random.randn(256, 256), cmap='gray')
            self.axes.set_title("")
            self.axes.axis('off')

        # Add buttons and label
        axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.21, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bprev = Button(axprev, 'Previous')
        self.bnext.on_clicked(self.next_set)
        self.bprev.on_clicked(self.prev_set)

        # Add a label to show the index of the set
        axlabel = plt.axes([0.32, 0.05, 0.2, 0.075])
        axlabel.axis('off')
        self.set_label = plt.text(0.42, 0.08, '', transform=axlabel.transAxes, ha='center', va='center')
         
        axbox = plt.axes([0.55, 0.05, 0.1, 0.075])
        self.text_box = TextBox(axbox, 'Index:')
        self.text_box.on_submit(self.jump_to_set)

        self.update_display()

        plt.show()

    def update_display(self):
        images_dict = self.process_fn(self.image_sets[self.current_set_index])

        if isinstance(self.axes, np.ndarray):
            for ax, imshow_return, key in zip(self.axes, self.imshow_return, images_dict):
                imshow_return.set_data(images_dict[key])
                imshow_return.set_clim(vmin=np.amin(images_dict[key]), vmax=np.amax(images_dict[key]) * 0.8)
                ax.set_title(key)
        else:
            for key in images_dict:
                self.imshow_return.set_data(images_dict[key])
                self.imshow_return.set_clim(vmin=np.amin(images_dict[key]), vmax=np.amax(images_dict[key]) * 0.8)
                self.axes.set_title(key)

        self.set_label.set_text(f'Index: [{self.current_set_index + 1} / {len(self.image_sets)}]')
        plt.draw()

    def next_set(self, event):
        if self.current_set_index < len(self.image_sets) - 1:
            self.current_set_index += 1
            self.update_display()

    def prev_set(self, event):
        if self.current_set_index > 0:
            self.current_set_index -= 1
            self.update_display()

    def jump_to_set(self, text):
        try:
            index = int(text) - 1
            if 0 <= index < len(self.image_sets):
                self.current_set_index = index
                self.update_display()
            else:
                print(f"Index out of range. Please enter a number between 1 and {len(self.image_sets)}.")

        except ValueError:
            print("Invalid input. Please enter a valid number.")


def generate_dataset_h5():

    my_seed = 0

    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)

    for idx_subset in ['tst_large']:
            for acceleration_rate in [4, 8]:

                FastBrainMRI(
                    idx_subset=idx_subset,
                    acceleration_rate=acceleration_rate,
                    datasheet=DATASHEET_DEFAULT,
                    is_return_y_smps_hat=False,
                    noise_sigma=0.01, 
                    mask_pattern='uniformly_cartesian_randomly_True',
                    smps_hat_method='eps',
                    num_coil_subset=None,
                    image_size_subset=None,
                    modality_subset=None,
                    is_verbose=True
                )


if __name__ == '__main__':

    dataset = FastBrainMRI(
        idx_subset='tra_medium',
        acceleration_rate=8,
        datasheet=DATASHEET_DEFAULT,
        is_return_y_smps_hat=True,
        noise_sigma=0.01, 
        mask_pattern='uniformly_cartesian_randomly_True',
        smps_hat_method='eps',
        num_coil_subset=20,
        image_size_subset=None,
        modality_subset="T2",
        is_verbose=True,
        is_pre_load=False
    )

    print(f"len(dataset): {len(dataset)}")

    """
    tra_large modality_subset=T2 image_size_subset=320 len of dataset: 13445
    val_large modality_subset=T2 image_size_subset=320 len of dataset: 1368
    tst_large modality_subset=T2 image_size_subset=320 len of dataset: 1338
    tra_large modality_subset=T1 image_size_subset=320 len of dataset: 2577
    val_large modality_subset=T1 image_size_subset=320 len of dataset: 273
    tst_large modality_subset=T1 image_size_subset=320 len of dataset: 261
    tra_large modality_subset=FLAIR image_size_subset=320 len of dataset: 2778
    val_large modality_subset=FLAIR image_size_subset=320 len of dataset: 291
    tst_large modality_subset=FLAIR image_size_subset=320 len of dataset: 291
    tra_large modality_subset=T1POST image_size_subset=320 len of dataset: 7283
    val_large modality_subset=T1POST image_size_subset=320 len of dataset: 723
    tst_large modality_subset=T1POST image_size_subset=320 len of dataset: 707
    tra_large modality_subset=T1PRE image_size_subset=320 len of dataset: 3271
    val_large modality_subset=T1PRE image_size_subset=320 len of dataset: 309
    tst_large modality_subset=T1PRE image_size_subset=320 len of dataset: 320
    tra_large modality_subset=T2 image_size_subset=396 len of dataset: 19336
    val_large modality_subset=T2 image_size_subset=396 len of dataset: 1922
    tst_large modality_subset=T2 image_size_subset=396 len of dataset: 1906

    tra_small modality_subset=T2 image_size_subset=320 len of dataset: 3064
    val_small modality_subset=T2 image_size_subset=320 len of dataset: 295
    tst_small modality_subset=T2 image_size_subset=320 len of dataset: 291
    tra_small modality_subset=T1 image_size_subset=320 len of dataset: 612
    val_small modality_subset=T1 image_size_subset=320 len of dataset: 64
    tst_small modality_subset=T1 image_size_subset=320 len of dataset: 49
    tra_small modality_subset=FLAIR image_size_subset=320 len of dataset: 606
    val_small modality_subset=FLAIR image_size_subset=320 len of dataset: 77
    tst_small modality_subset=FLAIR image_size_subset=320 len of dataset: 33
    tra_small modality_subset=T1POST image_size_subset=320 len of dataset: 1671
    val_small modality_subset=T1POST image_size_subset=320 len of dataset: 165
    tst_small modality_subset=T1POST image_size_subset=320 len of dataset: 165
    tra_small modality_subset=T1PRE image_size_subset=320 len of dataset: 674
    val_small modality_subset=T1PRE image_size_subset=320 len of dataset: 77
    tst_small modality_subset=T1PRE image_size_subset=320 len of dataset: 77
    tra_small modality_subset=T2 image_size_subset=396 len of dataset: 3796
    val_small modality_subset=T2 image_size_subset=396 len of dataset: 383
    tst_small modality_subset=T2 image_size_subset=396 len of dataset: 394
    """
