import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data
import csv
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register(suffix='basicsr')
class ShipSpottingDataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(ShipSpottingDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512
        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        # support multiple type of data: file path and meta data, remove support of lmdb
        self.paths = []
        if 'meta_info' in opt:
            with open(self.opt['meta_info']) as fin:
                    paths = [line.strip().split(' ')[0] for line in fin]
                    self.paths = [v for v in paths]
            if 'meta_num' in opt:
                self.paths = sorted(self.paths)[:opt['meta_num']]
        if 'gt_path' in opt:
            if isinstance(opt['gt_path'], str):
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path']).glob('*.'+opt['image_type'])]))
            else:
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][0]).glob('*.'+opt['image_type'])]))
                if len(opt['gt_path']) > 1:
                    for i in range(len(opt['gt_path'])-1):
                        self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]).glob('*.'+opt['image_type'])]))
        
        # limit number of pictures for test
        if 'num_pic' in opt:
            if 'val' or 'test' in opt:
                random.shuffle(self.paths)
                self.paths = self.paths[:opt['num_pic']]
            else:
                self.paths = self.paths[:opt['num_pic']]

        if 'mul_num' in opt:
            self.paths = self.paths * opt['mul_num']
            # print('>>>>>>>>>>>>>>>>>>>>>')
            # print(self.paths)

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        self.meta_info = get_all_csv_info_as_dict("/mnt/media/luigi/dataset/ShipSpotting/used")

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 10
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
            except Exception as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
                print("retrying", e)
            else:
                break
            finally:
                retry -= 1
        
        # img_gt = imfrombytes(img_bytes, float32=True)
        # filter the dataset and remove images with too low quality
        img_size = os.path.getsize(gt_path)
        img_size = img_size/1024

        while img_gt.shape[0] * img_gt.shape[1] < 384*384 or img_size<100:
            index = random.randint(0, self.__len__()-1)
            gt_path = self.paths[index]

            time.sleep(0.1)  # sleep 1s for occasional server congestion
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_size = os.path.getsize(gt_path)
            img_size = img_size/1024

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.crop_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            # top = (h - crop_pad_size) // 2 -1
            # left = (w - crop_pad_size) // 2 -1
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
        
        ship_name, ship_category, ship_type = self.get_vessel_name_and_category_and_vessel_type(index)
        text_prompt = self.create_sentence(ship_category, ship_name)
        return_d['text_prompt'] = text_prompt
        return return_d
    

    def create_sentence(self,category, ship_name):
        sentences = [
            "High resolution image of a ship named {} which is a majestic {}.",
            "High resolution image of {}, a well-known ship in the {} category.",
            "High resolution image of the ship {} which is quite popular among {} ships.",
            # "High resolution image of the ship {} named {} is truly a sight to behold.",
            "High resolution image of the {} which stands out among the {} ships."
        ]
        sentence = random.choice(sentences)
        return sentence.format(ship_name, category)

    def get_vessel_name_and_category_and_vessel_type(self, idx):
        # "3548158": {"image_title": "NARRABEEN & FAIRSTAR",
        #  "vessel_type": "Passengers Ship", 
        # "ship_category": "Passenger Vessels",
        #  "current_name": "NARRABEEN"}
        path_name = self.paths[idx]
        ship_type = self.meta_info[os.path.basename(path_name)[:-4]]['vessel_type']
        ship_name = self.meta_info[os.path.basename(path_name)[:-4]]['current_name']
        ship_category = self.meta_info[os.path.basename(path_name)[:-4]]['ship_category']
        return ship_name, ship_category, ship_type

    def __len__(self):
        return len(self.paths)


def extract_info_from_csv(csv_file, mini_dict):
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        # mini_dict = {}
        next(csv_reader)

        for row in csv_reader:
            mini_dict[row[0]] = {}
            # mini_dict = row[0]
            mini_dict[row[0]]['image_title'] = row[1]
            mini_dict[row[0]]['vessel_type'] = row[2]
            mini_dict[row[0]]['ship_category'] = row[3]
            mini_dict[row[0]]['current_name'] = row[4]
            line_count += 1

        # print(f'Processed {line_count} lines.')
        return mini_dict


#function to list all csv files in path and subpath
def list_all_csv_files(path):
    ls = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                ls.append(os.path.join(root, file))
    return ls

def get_all_csv_info_as_dict(path):
    csvs = list_all_csv_files(path)
    diz = {}
    for i in range(len(csvs)):
        diz = extract_info_from_csv(csvs[i], diz)
    return diz
    # print(len(diz))
    # print(diz['2988424'])