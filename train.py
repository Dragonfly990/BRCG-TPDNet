import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
import torch.nn.functional as F
from tqdm import tqdm
import random
from lib.Net import Net
from utils.dataloader import get_loader, test_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def train(train_loader, model, optimizer, epoch, opt):
    model.train()
    loss_record = AvgMeter()
    trainsize = opt.trainsize

    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
        gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

        # 模型输出3个分支 map0 map1 map2
        map_0, map_1, map_2 = model(images)

        loss0 = structure_loss(map_0, gts)
        loss1 = structure_loss(map_1, gts)
        loss2 = structure_loss(map_2, gts)
        loss = loss0 + loss1 + loss2

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        loss_record.update(loss.detach(), opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[total loss: {:.4f}]'.
                  format(datetime.now(), epoch + 1, opt.epoch, i, total_step,
                         loss_record.show()))


def val(model, opt):
    model.eval()
    val_img_root = f'./data/{opt.data}/Val/images/'
    val_gt_root = f'./data/{opt.data}/Val/masks/'
    val_loader = test_loader(val_img_root, val_gt_root, batchsize=1, trainsize=opt.trainsize)
    dice_sum = 0.0

    with torch.no_grad():
        for image, gt, _ in val_loader:
            image, gt = image.cuda(), gt.cuda()
            map0, map1, pred = model(image)
            pred = torch.sigmoid(pred)
            pred = F.interpolate(pred, size=gt.shape[2:], mode="bilinear", align_corners=False)
            pred = (pred > 0.5).float()

            inter = (pred * gt).sum()
            union = pred.sum() + gt.sum()
            dice = (2 * inter + 1) / (union + 1)
            dice_sum += dice.item()

    avg_dice = dice_sum / len(val_loader)
    model.train()
    return avg_dice

# ====================== 主程序入口 ======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CVC_ClinicDB', help='dataset folder name under data/')
    parser.add_argument('--epoch', type=int, default=100, help='total training epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--batchsize', type=int, default=6, help='train batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='input image resize size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping value')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='lr decay scale')
    parser.add_argument('--decay_epoch', type=int, default=20, help='lr decay interval epoch')
    parser.add_argument('--train_save', type=str, default='Net_Seg', help='save folder name in snapshots')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    model = Net().cuda()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    train_image_root = f'data/{opt.data}/train/images/'
    train_gt_root = f'data/{opt.data}/train/masks/'
    train_loader = get_loader(train_image_root, train_gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#" * 20, f"Start Training Dataset: {opt.data}", "#" * 20)
    best_dice = 0.0

    for epoch in range(0, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt)

        val_dice = val(model, opt)
        print(f"\n===== Epoch {epoch+1} Validation Result =====")
        print(f"Current Val Dice: {val_dice:.4f} | Best Val Dice: {best_dice:.4f}\n")

        save_dir = f"snapshots/{opt.train_save}/"
        os.makedirs(save_dir, exist_ok=True)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_dir + "Net-best.pth")
            print(f"New Best Model Saved! Best Dice Update To: {best_dice:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_dir + f"Net-{epoch}.pth")
            print(f"Saved checkpoint: Net-{epoch}.pth\n")
