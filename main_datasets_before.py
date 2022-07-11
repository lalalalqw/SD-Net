import os
import cv2
from torch.utils.data import Dataset
import numpy as np

# 篡改检测数据集 进行语义分割
class FluxDataset(Dataset):

    def __init__(self, dataset, mode):

        self.dataset = dataset
        self.mode = mode

        self.file_dir = './datasets/' + dataset + '/'#'G:/! CMFD datasets/USCISI-CMFD/' #

        f = open(self.file_dir + mode + '.txt')
        self.image_names = f.read().split('\n')
        self.image_names.pop()
        # self.image_names = self.image_names[-2:]

        self.dataset_length = len(self.image_names)

    def __len__(self):

        return self.dataset_length

    def __getitem__(self, index):

        image_name = self.image_names[index]

        # forgery image
        # image_dir = self.file_dir + self.mode + '_images/' + image_name
        image_dir = self.file_dir + 'images/' + image_name
        image = cv2.imread(image_dir)
        #########################
        # cv2.imwrite(self.file_dir + 'bigbigbig/' + image_name, image)
        # image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        # cv2.imwrite(self.file_dir + self.mode + '_images/' + image_name, image)
        #########################

        vis_image = image.copy()

        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)

        # mask image
        if self.dataset == 'CoMoFoD':
            mask_dir = self.file_dir + 'images/' + image_name[0:4] + 'B.png'
        elif self.dataset == 'GRIP++':
            mask_dir = self.file_dir + 'masks/' + image_name[0:-4] + '.png'
        elif self.dataset == 'GRIP':
            mask_dir = self.file_dir + 'images/' + image_name[0:-8] + 'gt.png'
        elif self.dataset == 'USCISI-CMFD':
            mask_dir = self.file_dir + self.mode + '_masks_0/' + image_name
        elif self.dataset == 'CASIA':
            mask_dir = self.file_dir + 'masks/' + image_name
        else:
            mask_dir = self.file_dir + 'forgery_mask/' + image_name

        label = cv2.imread(mask_dir, 0)
        #########################
        # cv2.imwrite(self.file_dir + 'bigbigbig_0/' + image_name, label)
        # label = cv2.resize(label, (0,0), fx=0.5, fy=0.5)
        # cv2.imwrite(self.file_dir + self.mode + '_masks_0/' + image_name, label)
        #########################
        # mask = label + 1
        mask = label.astype(np.float32)
        if self.dataset == 'GRIP':
            mask = label.astype(np.float32)

        return image, vis_image, mask, self.dataset_length, image_name

# sun_bcjxyczroqzvrngs-02-T0-X1279158.png
