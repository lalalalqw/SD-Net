import cv2
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import math

import matplotlib
# matplotlib.use('agg')
import pylab as plt

class FluxDataset(Dataset):

    def __init__(self, dataset, mode):

        self.dataset = dataset
        self.mode = mode

        self.file_dir = './datasets/' + dataset + '/'

        f = open(self.file_dir + mode + '.txt')
        self.image_names = f.read().split('\n')
        self.image_names.pop()

        self.dataset_length = len(self.image_names)

    def __len__(self):

        return self.dataset_length

    def __getitem__(self, index):

        image_name = self.image_names[index]

        # forgery image
        image_dir = self.file_dir + 'images/' + image_name
        image = cv2.imread(image_dir, 1)

        vis_image = image.copy()

        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)

        # seg image
        seg_dir = self.file_dir + 'seg_images/' + image_name[:-4] + '_seg.png'
        seg_im = cv2.imread(seg_dir, 0)
        seg_im = seg_im.astype(np.float32)
        seg_im = cv2.resize(seg_im, (int(math.ceil((math.ceil((seg_im.shape[1]-2)/2+1)-2)/2)+1),
                                     int(math.ceil((math.ceil((seg_im.shape[0]-2)/2+1)-2)/2)+1)))
        # mask image
        if self.dataset == 'CoMoFoD':
            mask_dir = self.file_dir + 'images/' + image_name[0:4] + 'B.png'
        if self.dataset == 'GRIP++':
            mask_dir = self.file_dir + 'masks/' + image_name[0:-4] + '.png'
        if self.dataset == 'GRIP':
            mask_dir = self.file_dir + 'images/' + image_name[0:-8] + 'gt.png'

        label = cv2.imread(mask_dir, 0)/255
        # mask = label + 1
        mask = label.astype(np.float32)
        mask_litte = cv2.resize(mask, (int(math.ceil((math.ceil((mask.shape[1]-2)/2+1)-2)/2)+1),
                                       int(math.ceil((math.ceil((mask.shape[0]-2)/2+1)-2)/2)+1)))
        if self.dataset == 'GRIP':
            mask = label.astype(np.float32)

        mat_dir = self.file_dir + 'seg_info/' + image_name[:-4] + '.mat'
        flux = sio.loadmat(mat_dir)['flux']
        norm = np.sqrt(flux[1, :, :] ** 2 + flux[0, :, :] ** 2)

        return image, seg_im, vis_image, mask, mask_litte, norm, self.dataset_length, image_name
