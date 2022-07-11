import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import scipy.io as sio
import torch
from select_vgg16_model import VGG16, VGG19, ResNet50, ResNet101, InceptionV3, DenseNet121
from main_vis_flux import vis_flux
from main_datasets_after import FluxDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

dataset = 'CoMoFoD'  # 'GRIP'  #
type = 'select_vgg16'
# select_vgg16  select_vgg19  select_ResNet50  select_ResNet101  select_InceptionV3  select_DenseNet121
test_save_dir = './test_saved/' + type
test_vis_dir = './test_vis/' + type + '/train/'  #'  #
mode = 'train'

def accuracy(pre, gt_mask):

    pre_01 = torch.round(pre)

    num = gt_mask.shape[0] * gt_mask.shape[1] * gt_mask.shape[2]

    correct = (pre_01 == gt_mask).sum()
    a = float(correct) / num

    zes = Variable(torch.zeros(gt_mask.shape).type(torch.LongTensor)).cuda()  # 全0变量
    ons = Variable(torch.ones(gt_mask.shape).type(torch.LongTensor)).cuda()  # 全1变量

    FN = ((pre_01 == zes) & (gt_mask.squeeze(1) == ons)).sum()  # train_correct01 原标签为1，预测为0 的总数
    FP = ((pre_01 == ons) & (gt_mask.squeeze(1) == zes)).sum()  # train_correct10 原标签为0，预测为1 的总数
    TP = ((pre_01 == ons) & (gt_mask.squeeze(1) == ons)).sum()  # train_correct11 原标签为1，预测为1 的总数
    TN = ((pre_01 == zes) & (gt_mask.squeeze(1) == zes)).sum()  # train_correct00 原标签为0，预测为0 的总数

    # p = float(TP) / float(TP + FP + 1e-6)
    # r = float(TP) / float(TP + FN + 1e-6)
    p = float(TN) / float(TN + FN + 1e-6)
    r = float(TN) / float(TN + FP + 1e-6)
    F = 2*p*r/(p + r + 1e-6)

    return a, p, r, F

def mean(l):
    sum = 0
    for i in l:
        sum = sum + i
    value = sum / len(l)
    return value

def main():

    if not os.path.exists(test_vis_dir + dataset):
        os.makedirs(test_vis_dir + dataset)

    if type == 'select_vgg16':
        model = VGG16()
    elif type == 'select_vgg19':
        model = VGG19()
    elif type == 'select_ResNet50':
        model = ResNet50()
    elif type == 'select_ResNet101':
        model = ResNet101()
    elif type == 'select_InceptionV3':
        model = InceptionV3()
    elif type == 'select_DenseNet121':
        model = DenseNet121()

    model = torch.nn.DataParallel(model)

    # model.load_state_dict(torch.load('./train_saved/' + type + '_' + dataset + '_400000.pth'))
    # model.load_state_dict(torch.load('./train_saved/final_CoMoFoD_400000.pth'))
    # model.load_state_dict(torch.load('./train_saved/no_seg_CoMoFoD_400000.pth'))
    # model.load_state_dict(torch.load('./train_saved/' + type + '_CoMoFoD_400000.pth'))
    model.load_state_dict(torch.load('./train_saved/' + type + '_CoMoFoD_100000.pth'))

    model.eval()
    model.cuda()

    dataloader = DataLoader(FluxDataset(dataset=dataset, mode=mode), batch_size=1, shuffle=False, num_workers=4)

    name, a, p, r, F = [], [], [], [], []

    for i_iter, batch_data in enumerate(dataloader):

        # 训练输入图像, 语义分割后图像，可视图像, mask, 数据集长度, 图片名
        Input_image, seg_im, vis_image, gt_mask, dataset_lendth, image_name = batch_data

        print('i_iter/total {}/{}'.format(i_iter, dataloader.__len__()))

        pred_flux = model(Input_image.cuda(), seg_im.cuda())

        # vis_flux(vis_image, pred_flux, gt_mask, image_name, test_vis_dir + dataset + '/')

        a_pred, p_pred, r_pred, F_pred = accuracy(pred_flux, gt_mask.cuda())
        name.append(image_name[0])
        a.append(a_pred)
        p.append(p_pred)
        r.append(r_pred)
        F.append(F_pred)

    sio.savemat(test_save_dir + '_' + dataset + '_results.mat', {'name': name, 'a': a, 'p': p, 'r': r, 'F': F})

    print('accuracy: {:.6f}'.format(mean(a)))
    print('p: {:.6f}'.format(mean(p)))
    print('r: {:.6f}'.format(mean(r)))
    print('F: {:.6f}'.format(mean(F)))

    # load = sio.loadmat(test_save_dir + '_' + dataset + '_results.mat')


if __name__ == '__main__':
    main()





