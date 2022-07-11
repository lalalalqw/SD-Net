import cv2
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import math


class TrainDataset(Dataset):

    def __init__(self, mode):

        self.mode = mode
        f_train = open('G:/! CMFD datasets/USCISI-CMFD/' + mode + '.txt')  #
        # f_train = open('./datasets/test2.txt')  # train
        self.image_names = f_train.read().split('\n')
        self.image_names.pop()

        self.dataset_length = len(self.image_names)

    def __len__(self):

        return self.dataset_length

    def __getitem__(self, index):

        ### forgery image
        image_name = self.image_names[index]
        # image_name = 'ADE20K/forgery_image/ADE_train_00009093.png'
        # image_dir = './datasets/' + image_name
        image_dir = 'G:/! CMFD datasets/USCISI-CMFD/' + self.mode + '_images/' + image_name
        image = cv2.imread(image_dir)
        image = cv2.resize(image, (512,512))####################

        # if (image.size > 1e6) or (image.size < 5e4):
        #     index = 0
        #     image_name = self.image_names[index]
        #     image_dir = './datasets/' + image_name
        #     image = cv2.imread(image_dir)

        vis_image = image.copy()

        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)

        ### seg image
        # seg_name = image_name.replace('forgery_image', 'seg_images')
        # seg_dir = './datasets/' + seg_name
        seg_dir = 'E:/CMFD/datasets/USCISI-CMFD/seg_images/' + image_name
        seg_im = cv2.imread(seg_dir, 0)
        seg_im = cv2.resize(seg_im, (512,512))#######################
        seg_im = seg_im.astype(np.float32)
        seg_im = cv2.resize(seg_im, (int(math.ceil((math.ceil((seg_im.shape[1]-2)/2+1)-2)/2)+1),
                                     int(math.ceil((math.ceil((seg_im.shape[0]-2)/2+1)-2)/2)+1)))

        ### mask image
        # mask_name = image_name.replace('forgery_image', 'forgery_mask')
        # mask_dir = './datasets/' + mask_name
        mask_dir = 'G:/! CMFD datasets/USCISI-CMFD/' + self.mode + '_masks_0/' + image_name
        mask = cv2.imread(mask_dir, 0)/255
        mask = cv2.resize(mask, (512,512))#####################
        # mask = mask + 1
        mask = mask.astype(np.float32)
        mask_litte = cv2.resize(mask, (int(math.ceil((math.ceil((mask.shape[1]-2)/2+1)-2)/2)+1),
                                       int(math.ceil((math.ceil((mask.shape[0]-2)/2+1)-2)/2)+1)))

        ### seg_mat
        mat_name = image_name.replace('forgery_image', 'seg_info').replace('.png', '.mat')
        mat_dir = 'E:/CMFD/datasets/USCISI-CMFD/seg_info/' + image_name[:-4] + '.mat'
        flux = sio.loadmat(mat_dir)['flux']
        norm = np.sqrt(flux[1, :, :] ** 2 + flux[0, :, :] ** 2)
        norm = cv2.resize(norm,(512,512))########################

        # ADE20K/forgery_image/ADE_train_00000001.png
        name = image_name.split('/')[-1]
        return image, seg_im, vis_image, mask, mask_litte, norm, self.dataset_length, name
        # return image, seg_im, vis_image, mask, mask_litte, self.dataset_length, name


class TestDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

        f_test = open('./datasets/' + self.dataset + '/image.txt')
        self.image_names = f_test.read().split('\n')
        self.image_names.pop()

        self.dataset_length = len(self.image_names)

    def __len__(self):

        return self.dataset_length

    def __getitem__(self, index):

        ### forgery image
        image_name = self.image_names[index]
        image_dir = './datasets/' + self.dataset + '/images/' + image_name
        image = cv2.imread(image_dir)
        image = cv2.resize(image, (512,512))############################

        vis_image = image.copy()

        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)

        ### seg image
        seg_dir = './datasets/' + self.dataset + '/seg_images/' + image_name[0:-4] + '_seg.png'
        if self.dataset == 'CASIA':
            seg_dir = './datasets/' + self.dataset + '/seg_images/' + image_name
        seg_im = cv2.imread(seg_dir, 0)
        seg_im = cv2.resize(seg_im, (512,512))#######################
        seg_im = seg_im.astype(np.float32)
        seg_im = cv2.resize(seg_im, (int(math.ceil((math.ceil((seg_im.shape[1]-2)/2+1)-2)/2)+1),
                                     int(math.ceil((math.ceil((seg_im.shape[0]-2)/2+1)-2)/2)+1)))

        ### seg_mat
        mat_dir = './datasets/' + self.dataset + '/seg_info/' + image_name[0:-4] + '.mat'
        flux = sio.loadmat(mat_dir)['flux']
        norm = np.sqrt(flux[1, :, :] ** 2 + flux[0, :, :] ** 2)
        norm = cv2.resize(norm,(512,512))########################

        ### mask image
        if self.dataset == 'CoMoFoD':
            mask_dir = './datasets/CoMoFoD/images/' + image_name[0:4] + 'B.png'
            mask = cv2.imread(mask_dir, 0)/255
            mask[:, -1] = 0
        elif self.dataset == 'CASIA':
            mask_dir = './datasets/CASIA/masks/' + image_name
            mask = cv2.imread(mask_dir, 0)/255
        elif self.dataset == 'GRIP':
            mask_dir = './datasets/GRIP/images/' + image_name[0:-8] + 'gt.png'
            mask = cv2.imread(mask_dir, 0) / 255

        mask = cv2.resize(mask, (512,512))#######################

        mask = mask.astype(np.float32)
        mask_litte = cv2.resize(mask, (int(math.ceil((math.ceil((mask.shape[1]-2)/2+1)-2)/2)+1),
                                       int(math.ceil((math.ceil((mask.shape[0]-2)/2+1)-2)/2)+1)))
        # return image, seg_im, vis_image, mask, mask_litte, self.dataset_length, image_name
        return image, seg_im, vis_image, mask, mask_litte, norm, self.dataset_length, image_name


class TrainDatasetCMFD(Dataset):

    def __init__(self):

        f_train = open('./datasets/test2.txt')  # train
        self.image_names = f_train.read().split('\n')
        self.image_names.pop()

        self.dataset_length = len(self.image_names)

    def __len__(self):

        return self.dataset_length

    def __getitem__(self, index):

        ### forgery image
        image_name = self.image_names[index]
        # image_name = 'ADE20K/forgery_image/ADE_train_00009093.png'
        image_dir = './datasets/' + image_name
        image = cv2.imread(image_dir)
        image = cv2.resize(image, (512,512))############################

        # if (image.size > 1e6) or (image.size < 5e4):
        #     index = 0
        #     image_name = self.image_names[index]
        #     image_dir = './datasets/' + image_name
        #     image = cv2.imread(image_dir)

        vis_image = image.copy()

        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)

        ### seg image
        seg_name = image_name.replace('forgery_image', 'seg_images')
        seg_dir = './datasets/' + seg_name
        seg_im = cv2.imread(seg_dir, 0)
        seg_im = cv2.resize(seg_im, (512,512))#######################
        seg_im = seg_im.astype(np.float32)
        seg_im = cv2.resize(seg_im, (int(math.ceil((math.ceil((seg_im.shape[1]-2)/2+1)-2)/2)+1),
                                     int(math.ceil((math.ceil((seg_im.shape[0]-2)/2+1)-2)/2)+1)))

        ### mask image
        mask_name = image_name.replace('forgery_image', 'forgery_mask')
        mask_dir = './datasets/' + mask_name
        mask = cv2.imread(mask_dir, 0)/255
        mask = cv2.resize(mask, (512,512))#####################
        # mask = mask + 1
        mask = mask.astype(np.float32)
        mask_litte = cv2.resize(mask, (int(math.ceil((math.ceil((mask.shape[1]-2)/2+1)-2)/2)+1),
                                       int(math.ceil((math.ceil((mask.shape[0]-2)/2+1)-2)/2)+1)))

        ### seg_mat
        mat_name = image_name.replace('forgery_image', 'seg_info').replace('.png', '.mat')
        mat_dir = './datasets/' + mat_name
        flux = sio.loadmat(mat_dir)['flux']
        norm = np.sqrt(flux[1, :, :] ** 2 + flux[0, :, :] ** 2)
        norm = cv2.resize(norm,(512,512))########################

        # ADE20K/forgery_image/ADE_train_00000001.png
        name = image_name.split('/')[-1]
        return image, seg_im, vis_image, mask, mask_litte, norm, self.dataset_length, name