import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import torch
import torch.nn as nn
from main_model import VGG16
from main_vis_flux import vis_flux
from main_datasets_after import FluxDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# import torchsummary
from torchinfo import summary

INI_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
EPOCHES = 1000

dataset = 'CoMoFoD'
type = 'final_SelfCorr'
snapshot_dir = './train_saved/'
train_vis_dir = './train_vis/' + type + '/'
mode = 'train'

# def loss_calc(pred_flux, gt_flux):
#
#     device_id = pred_flux.device
#     gt_flux = gt_flux.cuda(device_id)
#
#     pred_flux = pred_flux[:, 0, :, :]
#
#     gt_flux = 0.999999 * gt_flux / (gt_flux.norm(p=2, dim=1) + 1e-9)
#
#     # norm loss
#     norm_loss = (pred_flux - gt_flux)**2
#     norm_loss = norm_loss.sum()
#
#     # angle loss
#     pred_flux = 0.999999 * pred_flux / (pred_flux.norm(p=2, dim=1) + 1e-9)
#
#     angle_loss = (torch.acos(torch.sum(pred_flux * gt_flux, dim=1)))**2
#     angle_loss = angle_loss.sum()
#
#     return norm_loss, angle_loss

    # bce_loss = torch.sigmoid(-gt_flux*torch.log(pred_flux) - (1-gt_flux)*torch.log(pred_flux))
    # bce_loss = bce_loss.sum()
    # gt1 = gt_flux.view(-1)
    # pred1 = pred_flux.view(-1)
    # smooth = 1
    # bce_loss = F.binary_cross_entropy(pred1, gt1, reduction='mean')
    # intersection = (pred1 * gt1).sum()
    # total = (pred1 + gt1).sum()
    # union = total - intersection
    # IoU = (intersection + smooth) / (union + smooth)
    # iou_loss = 1 - IoU
    # return bce_loss, iou_loss

def get_params(model, key, bias=False):
    # for backbone 
    if key == "backbone":
        for m in model.named_modules():
            if "backbone" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    if not bias:
                        yield m[1].weight
                    else:
                        yield m[1].bias
    # for added layer
    if key == "added":
        for m in model.named_modules():
            if "backbone" not in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    if not bias:
                        yield m[1].weight
                    else:
                        yield m[1].bias

def adjust_learning_rate(optimizer, step):
    
    if step == 8e4:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

# def acc(pred_flux,gt_mask):
#     gt_mask = gt_mask.data.cpu().numpy()[0, ...]
#     pred_flux = pred_flux.data.cpu().numpy()[0, ...]
#
#     zes = Variable(torch.zeros(gt_mask.size()).type(torch.LongTensor))  # 全0变量
#     ons = Variable(torch.ones(gt_mask.size()).type(torch.LongTensor))  # 全1变量
#     train_correct01 = ((pred_flux == zes) & (gt_mask.squeeze(1) == ons)).sum()  # 原标签为1，预测为 0 的总数
#     train_correct10 = ((pred_flux == ons) & (gt_mask.squeeze(1) == zes)).sum()  # 原标签为0，预测为1  的总数
#     train_correct11 = ((pred_flux == ons) & (gt_mask.squeeze(1) == ons)).sum()
#     train_correct00 = ((pred_flux == zes) & (gt_mask.squeeze(1) == zes)).sum()
#
#     FN,FP,TP,TN = 0
#
#     FN += train_correct01.data[0]
#     FP += train_correct10.data[0]
#     TP += train_correct11.data[0]
#     TN += train_correct00.data[0]
#
#     P = TP / (TP + FP)
#     R = TP / (TP + FN)
#
#     F = 2*P*R/(P+R)
#
#     return P,R,F

def loss(pred_flux, gt_mask):
    device_id = pred_flux.device
    gt_mask = gt_mask.cuda(device_id)
    pred_flux = pred_flux[0, :, :]

    # total_loss = gt_mask * torch.log(pred_flux) + (1 - gt_mask) * torch.log(1 - pred_flux)
    total_loss = (pred_flux - gt_mask)**2
    total_loss = total_loss.sum() / (gt_mask.shape[1] * gt_mask.shape[2])

    return total_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    # if not os.path.exists(snapshot_dir):
    #     os.makedirs(snapshot_dir)
    # if not os.path.exists(train_vis_dir + dataset):
    #     os.makedirs(train_vis_dir + dataset)

    model = VGG16()

    # 计算参数
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))

    summary(model, input_size=[(1,3, 512, 512), (1,128, 128), (1,512, 512)])

    # model = torchvision.models.vgg16(pretrained=True)

    # writer = SummaryWriter('loss/model_vgg16')
    model = nn.DataParallel(model)

    saved_dict = torch.load('vgg16_pretrain.pth')
    model_dict = model.state_dict()
    saved_key = list(saved_dict.keys())
    model_key = list(model_dict.keys())

    for i in range(1, 26):
        model_dict[model_key[i]] = saved_dict[saved_key[i]]

    model.load_state_dict(model_dict)

    model.train()
    model.cuda()
    loss = nn.BCEWithLogitsLoss() #nn.BCELoss() #

    optimizer = torch.optim.Adam(
        params=[
            {
                "params": get_params(model, key="backbone", bias=False),
                "lr": INI_LEARNING_RATE
            },
            {
                "params": get_params(model, key="backbone", bias=True),
                "lr": 2 * INI_LEARNING_RATE
            },
            {
                "params": get_params(model, key="added", bias=False),
                "lr": 10 * INI_LEARNING_RATE  
            },
            {
                "params": get_params(model, key="added", bias=True),
                "lr": 20 * INI_LEARNING_RATE   
            },
        ],
        weight_decay=WEIGHT_DECAY
    )

    dataloader = DataLoader(FluxDataset(dataset=dataset, mode=mode), batch_size=1, shuffle=True, num_workers=4)

    global_step = 0

    for epoch in range(1, EPOCHES):

        for i_iter, batch_data in enumerate(dataloader):

            global_step += 1

            # 训练输入图像, 语义分割后图像，可视图像, mask, 数据集长度, 图片名
            Input_image, seg_im, vis_image, gt_mask, dataset_lendth, image_name = batch_data

            optimizer.zero_grad()

            pred_flux = model(Input_image.cuda(), seg_im.cuda())

            total_loss = loss(pred_flux, gt_mask.cuda())
            total_loss.backward()

            optimizer.step()

            # if global_step % 100 == 0:
            print('epoche {} i_iter/total {}/{} loss {:.6f}'.format(epoch, i_iter, int(dataset_lendth.data),total_loss))
            # writer.add_scalar('loss', total_loss, i_iter + epoch * int(dataset_lendth.data))
            # writer.add_scalar('p', p, i_iter + epoch * int(dataset_lendth.data))
            # writer.add_scalar('r', r, i_iter + epoch * int(dataset_lendth.data))
            # writer.add_scalar('F', F, i_iter + epoch * int(dataset_lendth.data))
                # writer.add_scalar('norm_loss', norm_loss, i_iter)
                # writer.add_scalar('angle_loss', angle_loss, i_iter)

            if global_step % 500 == 0:
                vis_flux(vis_image, pred_flux, gt_mask, image_name, train_vis_dir + dataset + '/')

            if global_step % 1e4 == 0:
                torch.save(model.state_dict(), snapshot_dir + type + '_' + dataset + '_' + str(global_step) + '.pth')

            if global_step % 4e6 == 0:
                return
    # writer.close()
    # tensorboard --logdir=./model_all
    # http://localhost:6006/


if __name__ == '__main__':
    main()





