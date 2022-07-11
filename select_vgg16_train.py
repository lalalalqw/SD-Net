from torchvision import datasets, models, transforms
model = models.densenet121(pretrained=True)


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import torch
import torch.nn as nn
from select_vgg16_model import VGG16, VGG19, ResNet50, ResNet101, InceptionV3, DenseNet121
from main_vis_flux import vis_flux
from main_datasets_after import FluxDataset
from torch.utils.data import DataLoader
# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pylab as plt

INI_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
EPOCHES = 100

dataset = 'CoMoFoD'
type = 'select_DenseNet121'
# select_vgg16  select_vgg19  select_ResNet50  select_ResNet101  select_InceptionV3  select_DenseNet121
snapshot_dir = './train_saved/'
train_vis_dir = './train_vis/' + type + '/'
mode = 'train'

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

# def loss(pred_flux, gt_mask):
#     device_id = pred_flux.device
#     gt_mask = gt_mask.cuda(device_id)
#     pred_flux = pred_flux[0, :, :]
#
#     # total_loss = gt_mask * torch.log(pred_flux) + (1 - gt_mask) * torch.log(1 - pred_flux)
#     total_loss = (pred_flux - gt_mask)**2
#     total_loss = total_loss.sum() / (gt_mask.shape[1] * gt_mask.shape[2])
#
#     return total_loss

def main():

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    if not os.path.exists(train_vis_dir + dataset):
        os.makedirs(train_vis_dir + dataset)

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

    model = nn.DataParallel(model)

    model.train()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    # optimizer = torch.optim.Adam(
    #     params=[
    #         {
    #             "params": get_params(model, key="backbone", bias=False),
    #             "lr": INI_LEARNING_RATE
    #         },
    #         {
    #             "params": get_params(model, key="backbone", bias=True),
    #             "lr": 2 * INI_LEARNING_RATE
    #         },
    #         {
    #             "params": get_params(model, key="added", bias=False),
    #             "lr": 10 * INI_LEARNING_RATE
    #         },
    #         {
    #             "params": get_params(model, key="added", bias=True),
    #             "lr": 20 * INI_LEARNING_RATE
    #         },
    #     ],
    #     weight_decay=WEIGHT_DECAY
    # )
    loss = nn.BCEWithLogitsLoss()

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

            print('epoche {} i_iter/total {}/{} loss {:.6f}'.format(epoch, i_iter, int(dataset_lendth.data),total_loss))

            if global_step % 500 == 0:
                vis_flux(vis_image, pred_flux, gt_mask, image_name, train_vis_dir + dataset + '/')

            if global_step % 1e4 == 0:
                torch.save(model.state_dict(), snapshot_dir + type + '_' + dataset + '_' + str(global_step) + '.pth')

            # if global_step % 4e6 == 0:
            #     return
    # tensorboard --logdir=./select_vgg16_loss/model_vgg16
    # http://localhost:6006/


if __name__ == '__main__':
    main()





