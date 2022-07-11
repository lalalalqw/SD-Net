import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from main_model import VGG16
from main_vis_flux import vis_flux
from train_datasets import TrainDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

INI_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
EPOCHES = 1000

data = '0619'
snapshot_dir = './train_saved/' + data + '/'
train_vis_dir = './train_vis/' + data + '/'
loss_dir = 'loss/train' + data + '2'  # tensorboard --logdir=./loss/train05


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
    if step == 8e5:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def main():
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    if not os.path.exists(train_vis_dir):
        os.makedirs(train_vis_dir)

    model = VGG16()

    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load('./train_saved/0512/340000.pth'))

    model.train()
    model.cuda()
    loss = nn.BCEWithLogitsLoss()  #nn.BCELoss()  #

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

    dataloader = DataLoader(TrainDataset(mode='train'), batch_size=4, shuffle=True, num_workers=8)

    global_step = 0
    writer = SummaryWriter(loss_dir)
    for epoch in range(1, EPOCHES):

        for i_iter, batch_data in enumerate(dataloader):

            global_step += 1

            # 训练输入图像, 语义分割后图像，可视图像, mask, 边缘, 数据集长度, 图片名
            Input_image, seg_im, vis_image, gt_mask, mask_litte, norm, dataset_lendth, image_name = batch_data
            # Input_image, seg_im, vis_image, gt_mask, mask_litte, dataset_lendth, image_name = batch_data

            optimizer.zero_grad()

            pred_flux = model(Input_image.cuda(), seg_im.cuda(), norm.cuda())

            total_loss = loss(pred_flux, gt_mask.cuda())
            total_loss.backward()

            # acc = acce(pred_flux, gt_mask.cuda())

            optimizer.step()

            # torch.cuda.empty_cache() ########### 清显存

            if global_step % 100 == 0:
                print('epoche {} i_iter/total {}/{} loss {:.6f}'.
                      format(epoch, i_iter, dataloader.__len__(), total_loss.item()))

            writer.add_scalar('loss', total_loss.item(), i_iter + epoch * dataloader.__len__())

            if global_step % 500 == 0:
                vis_flux(vis_image, pred_flux, gt_mask, image_name, train_vis_dir)

            if global_step % 1e4 == 0:
                torch.save(model.state_dict(), snapshot_dir + str(global_step) + '.pth')

            # if global_step % 4e6 == 0:
            #     return
    writer.close()


if __name__ == '__main__':
    main()
