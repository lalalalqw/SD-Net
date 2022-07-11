import matplotlib
matplotlib.use('agg')
import pylab as plt
plt.rc('font', family='Times New Roman')

def vis_flux(vis_image, pred_flux, gt_mask, image_name, save_dir):

    vis_image = vis_image.data.cpu().numpy()[0, ...]
    pred_flux = pred_flux.data.cpu().numpy()[0, ...]
    gt_mask = gt_mask.data.cpu().numpy()[0, ...]

    image_name = image_name[0]

    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(131)
    ax1.imshow(vis_image[:, :, ::-1])
    ax1.set_title('image')
    plt.axis('off')

    ax2 = fig.add_subplot(132)
    ax2.imshow(gt_mask)
    ax2.set_title('gt_mask')
    plt.axis('off')

    ax3 = fig.add_subplot(133)
    ax3.imshow(pred_flux)
    ax3.set_title('pred_flux')
    plt.axis('off')

    plt.savefig(save_dir + image_name[0:-4] + '.png')
    plt.imsave(save_dir + image_name[0:-4] + '_pred.png', pred_flux)
    plt.close(fig)
