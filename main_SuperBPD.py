import os
import torch
from SuperBPD_model import VGG16
from main_datasets_before import FluxDataset
import scipy.io as sio
from torch.utils.data import DataLoader
import concurrent.futures

import bpd_cuda
import math
import cv2
import numpy as np
from torchinfo import summary

import matplotlib
matplotlib.use('agg')
import pylab as plt
from matplotlib import cm
plt.rc('font', family='Times New Roman')

dataset = 'CASIA'
test_vis_dir = './datasets/' + dataset
mode = 'test'#

################# 填色
def label2color(label):
    label = label.astype(np.uint16)

    height, width = label.shape
    color3u = np.zeros((height, width, 3), dtype=np.uint8)
    unique_labels = np.unique(label)

    if unique_labels[-1] >= 2 ** 24:
        raise RuntimeError('Error: label overflow!')

    for i in range(len(unique_labels)):
        binary = '{:024b}'.format(unique_labels[i])
        # r g b 3*8 24
        r = int(binary[::3][::-1], 2)
        g = int(binary[1::3][::-1], 2)
        b = int(binary[2::3][::-1], 2)

        color3u[label == unique_labels[i]] = np.array([r, g, b])

    return color3u

################# 后处理
def post_process(flux, name):
    flux = torch.from_numpy(flux).cuda()

    angles = torch.atan2(flux[1, ...], flux[0, ...])
    angles[angles < 0] += 2 * math.pi

    height, width = angles.shape

    # theta_a, theta_l, theta_s, S_o, 45, 116, 68, 5
    results = bpd_cuda.forward(angles, height, width, 45, 116, 68, 5)
    root_points, super_BPDs_before_dilation, super_BPDs_after_dilation, super_BPDs = results

    super_BPDs = super_BPDs.cpu().numpy()
    super_BPDs = label2color(super_BPDs)
    cv2.imwrite(name, super_BPDs)

    # root_points = root_points.cpu().numpy()
    # root_points = 255 * (root_points > 0)
    #
    # return root_points, super_BPDs

################# 过程显示
def vis_flux(vis_image, pred_flux, root_points, super_BPDs, gt_mask, save_name):

    norm_pred = np.sqrt(pred_flux[1, :, :] ** 2 + pred_flux[0, :, :] ** 2)
    angle_pred = 180 / math.pi * np.arctan2(pred_flux[1, :, :], pred_flux[0, :, :])

    fig = plt.figure(figsize=(10, 6))

    ax0 = fig.add_subplot(231)
    ax0.set_title('forgery image')
    ax0.imshow(vis_image[:, :, ::-1])
    plt.axis('off')

    ax1 = fig.add_subplot(232)
    ax1.set_title('Norm pred')
    ax1.set_autoscale_on(True)
    im1 = ax1.imshow(norm_pred, cmap=cm.jet)
    plt.colorbar(im1, shrink=0.5)
    plt.axis('off')

    ax2 = fig.add_subplot(233)
    ax2.set_title('Angle pred')
    ax2.set_autoscale_on(True)
    im2 = ax2.imshow(angle_pred, cmap=cm.jet)
    plt.colorbar(im2, shrink=0.5)
    plt.axis('off')

    ax3 = fig.add_subplot(234)
    ax3.set_title('root')
    ax3.imshow(root_points, cmap='gray')
    plt.axis('off')

    ax4 = fig.add_subplot(235)
    ax4.set_title('super BPDs')
    ax4.imshow(super_BPDs)
    plt.axis('off')

    ax5 = fig.add_subplot(236)
    ax5.set_title('mask')
    # gt_mask = abs(gt_mask - 1)
    ax5.imshow(gt_mask, cmap='gray')
    plt.axis('off')

    plt.savefig(save_name)
    plt.close(fig)

def main():

    # if not os.path.exists(test_vis_dir + '/seg_images/'):
    #     os.makedirs(test_vis_dir + '/seg_images/')
    #     os.makedirs(test_vis_dir + '/seg_info/')

    model = VGG16()
    # 计算参数
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
    summary(model, input_size=[(1, 3, 512, 512)])

    model.load_state_dict(torch.load('./SuperBPD_saved/PascalContext_400000.pth'))

    model.eval()
    model.cuda()
    
    dataloader = DataLoader(FluxDataset(dataset=dataset, mode=mode), batch_size=1, shuffle=False, num_workers=4)
    # fh = open("testfileval.txt", "w", encoding='UTF-8')
    for i_iter, batch_data in enumerate(dataloader):
        #训练输入图像, 可视图像, mask, 数据集长度, 图片名
        Input_image, vis_image, gt_mask, dataset_lendth, image_name = batch_data

        print(i_iter, dataset_lendth, image_name[0])

        # try:
        pred_flux = model(Input_image.cuda())  #
        # except:
        #     fh.write(image_name[0] + '\n')
        #     continue
        pred_flux = pred_flux.data.cpu().numpy()[0, ...]

        post_process(pred_flux, test_vis_dir + '/seg_images/' + image_name[0])
        # vis_flux(vis_image, pred_flux, root_points, super_BPDs, gt_mask, test_vis_dir + '/seg_images/' + image_name[0][0:-4] + '.mat')

        sio.savemat(test_vis_dir + '/seg_info/' + image_name[0][0:-4] + '.mat', {'flux': pred_flux})
    # fh.close()

def show(image_name):
    flux = sio.loadmat(test_vis_dir + '/seg_info/' + image_name[0:-4] + '.mat')['flux']
    # root_points, super_BPDs =
    post_process(flux, test_vis_dir + '/seg_images/' + image_name[0:-4] + '.png')

    # vis_image = cv2.imread(test_vis_dir + '/forgery_image/' + image_name)
    # gt_mask = cv2.imread(test_vis_dir + '/forgery_mask/' + image_name, 0)
    # vis_flux(vis_image, flux, root_points, super_BPDs, gt_mask,
    #          test_vis_dir + '/seg_info/' + image_name[0:-4] + '.png')

    print(image_name)

def main_after():
    f_train = open('./datasets/' + dataset + '/train.txt')
    image_names = f_train.read().split('\n')
    image_names.pop()
    image_names = image_names[16770:]
    # show('ADE_train_00006115.png')

    for image_name in image_names:
        show(image_name)
    # 并行加速 -- 内存会爆
    # with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
    #     image_name = image_names
    #     executor.map(show, image_name)

if __name__ == '__main__':
    main()
    # main_after()

