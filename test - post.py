import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import scipy.io as sio
import os
import torch
import torch.nn as nn
from main_model import VGG16, Post
from main_vis_flux import vis_flux
from train_datasets import TestDataset#TrainDataset#
from torch.utils.data import DataLoader
from torch.autograd import Variable

dataset = 'CoMoFoD'#CoMoFoD  USCISI-CMFD  CASIA  GRIP
test_save_dir = './test_saved/0602_'
test_vis_dir = './test_vis/0602_' + dataset + '/'#train/'

def accuracy(pre, gt_mask):

    pre_01 = torch.round(pre)

    num = gt_mask.shape[0] * gt_mask.shape[1] * gt_mask.shape[2]

    correct = (pre_01 == gt_mask).sum()
    a = float(correct) / num

    zes = torch.zeros(gt_mask.shape).long().cuda()  # 全0变量
    ons = torch.ones(gt_mask.shape).long().cuda()  # 全1变量

    FN = ((pre_01 == zes) & (gt_mask.squeeze(1) == ons)).sum()  # train_correct01 原标签为1，预测为 0 的总数
    FP = ((pre_01 == ons) & (gt_mask.squeeze(1) == zes)).sum()  # train_correct10 原标签为0，预测为1  的总数
    TP = ((pre_01 == ons) & (gt_mask.squeeze(1) == ons)).sum()  # train_correct11
    TN = ((pre_01 == zes) & (gt_mask.squeeze(1) == zes)).sum()  # train_correct00

    p = float(TP) / float(TP + FP + 1e-6)
    r = float(TP) / float(TP + FN + 1e-6)
    F = 2 * p * r / (p + r + 1e-6)

    return a, p, r, F
def mean(l):
    sum = 0
    for i in l:
        sum = sum + i
    value = sum / len(l)
    return value
def main():

    if not os.path.exists(test_vis_dir):
        os.makedirs(test_vis_dir)

    model0 = VGG16()
    model0 = nn.DataParallel(model0)
    model0.load_state_dict(torch.load('./train_saved/0525/190000.pth'))
    model0.eval()
    model0.cuda()

    model = Post()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./train_saved/0529/40000.pth'))
    model.eval()
    model.cuda()

    # TrainDataset(mode='train')TestDataset(dataset=dataset)
    dataloader = DataLoader(TestDataset(dataset=dataset), batch_size=1, shuffle=False, num_workers=8)
    name, a, p, r, F = [], [], [], [], []

    for i_iter, batch_data in enumerate(dataloader):

        # 训练输入图像, 语义分割后图像，可视图像, mask, 边缘, 数据集长度, 图片名
        # Input_image, seg_im, vis_image, gt_mask, norm, dataset_lendth, image_name = batch_data
        Input_image, seg_im, vis_image, gt_mask, mask_litte, norm, dataset_lendth, image_name = batch_data
        # Input_image, seg_im, vis_image, gt_mask, mask_litte, dataset_lendth, image_name = batch_data

        print('i_iter/total {}/{}'.format(i_iter, dataloader.__len__()))

        # pred_flux = model(Input_image.cuda(), seg_im.cuda(), norm.cuda())
        pred_flux0 = model0(Input_image.cuda(), seg_im.cuda())
        pred_flux = model(pred_flux0, norm.cuda())

        pred_flux = nn.Sigmoid()(pred_flux)

        vis_flux(vis_image, pred_flux, gt_mask, image_name, test_vis_dir)

        a_pred, p_pred, r_pred, F_pred = accuracy(pred_flux, gt_mask.cuda())
        name.append(image_name[0])
        a.append(a_pred)
        p.append(p_pred)
        r.append(r_pred)
        F.append(F_pred)

    sio.savemat(test_save_dir + dataset + '_results.mat', {'name': name, 'a': a, 'p': p, 'r': r, 'F': F})

    print('accuracy: {:.6f}'.format(mean(a)))
    print('p: {:.6f}'.format(mean(p)))
    print('r: {:.6f}'.format(mean(r)))
    print('F: {:.6f}'.format(mean(F)))

    # load = sio.loadmat(test_save_dir + '_' + dataset + '_results.mat')


if __name__ == '__main__':
    main()





