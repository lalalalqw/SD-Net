import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import torch
import torch.nn as nn
from no_seg_model import VGG16
from main_vis_flux import vis_flux
from main_datasets_before import FluxDataset
from torch.utils.data import DataLoader

INI_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
EPOCHES = 1000

dataset = 'CoMoFoD'
type = 'no_seg'
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

def loss(pred_flux, gt_mask):
    device_id = pred_flux.device
    gt_mask = gt_mask.cuda(device_id)
    pred_flux = pred_flux[0, :, :]

    # total_loss = gt_mask * torch.log(pred_flux) + (1 - gt_mask) * torch.log(1 - pred_flux)
    total_loss = (pred_flux - gt_mask)**2
    total_loss = total_loss.sum() / (gt_mask.shape[1] * gt_mask.shape[2])

    return total_loss

def main():

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    if not os.path.exists(train_vis_dir + dataset):
        os.makedirs(train_vis_dir + dataset)

    model = VGG16()

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
    # loss = nn.BCEWithLogitsLoss() #nn.BCELoss() #
    
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

            # 训练输入图像, 可视图像, mask, 数据集长度, 图片名
            Input_image, vis_image, gt_mask, dataset_lendth, image_name = batch_data

            optimizer.zero_grad()

            pred_flux = model(Input_image.cuda())

            total_loss = loss(pred_flux, gt_mask.cuda())
            total_loss.backward()

            optimizer.step()

            # if global_step % 100 == 0:
            print('epoche {} i_iter/total {}/{} loss {:.6f}'.format(epoch, i_iter, int(dataset_lendth.data),total_loss))

            if global_step % 500 == 0:
                vis_flux(vis_image, pred_flux, gt_mask, image_name, train_vis_dir + dataset + '/')

            if global_step % 1e4 == 0:
                torch.save(model.state_dict(), snapshot_dir +  type + '_' + dataset + '_' + str(global_step) + '.pth')

            if global_step % 4e6 == 0:
                return


if __name__ == '__main__':
    main()

