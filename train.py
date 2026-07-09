import torch
import torch.nn as nn
import os
import numpy as np
import argparse

from tqdm import tqdm
import random

import torch.nn.functional as F

from dataloader import split_with_val, get_loader, ValDataset
from utils import AvgMeter, adjust_lr
from val import val
from Net import Net


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        import cv2
        cv2.setRNGSeed(seed)
    except ImportError:
        pass


class FocalLossV1(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        logits = logits.float()
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha
        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        return loss.mean()

def structure_loss1(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wfocal = FocalLossV1()(pred, mask)
    wfocal = (wfocal * weit).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wfocal + wiou).mean()

# ===================== Training Function =====================
def train(train_loader, model, optimizer, opt, epoch):
    model.train()
    loss_record = AvgMeter()
    
    for i, pack in tqdm(enumerate(train_loader), total=len(train_loader)):
        image, mask = pack
        image, mask = image.cuda(), mask.cuda()
        optimizer.zero_grad()
        
        map_0, map_1, map_2 = model(image)
        
        loss0 = structure_loss1(map_0, mask)
        loss1 = structure_loss1(map_1, mask)
        loss2 = structure_loss1(map_2, mask)
        loss = loss0 + loss1 + loss2
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.clip)
        optimizer.step()
        loss_record.update(loss.data, opt.batchsize)
    
    print('[total loss: {:.4f}]'.format(loss_record.show()))
    return loss_record.show()

# ===================== Main =====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CVC-ClinicDB')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=6)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--decay_epoch', type=int, default=30)
    parser.add_argument('--train_save', type=str, default='Net')
    parser.add_argument('--val_list', type=str, default='data/CVC-ClinicDB/val_list.txt')
    parser.add_argument('--seed', type=int, default=212)
    opt = parser.parse_args()

    full_train_img_root = f'data/{opt.data}/train/images/'
    full_train_gt_root = f'data/{opt.data}/train/masks/'
    test_image_root = f'./data/{opt.data}/test/images/'
    test_gt_root = f'./data/{opt.data}/test/masks/'

    set_seed(opt.seed)

    opt.model = f'Net'

    model = Net().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    train_img_list, train_mask_list, val_img_list, val_mask_list = split_with_val(
            full_train_img_root, full_train_gt_root, opt.val_list
        )

    train_loader = get_loader(train_img_list, train_mask_list, opt.batchsize, opt.size, seed=opt.seed)

    model_path = f'snapshots/{opt.data}/{opt.train_save}/'
    os.makedirs(model_path, exist_ok=True)
   
  
    print("#" * 20, f"Start Training round", "#" * 20)

    train_pro = np.zeros((opt.epoch, 5))
    best_dice = 0.0
    best_iou = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_epoch = 0

    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
          
        print('*' * 10, f'round, epoch {epoch}', '*' * 10)
            
        loss = train(train_loader, model, optimizer, opt, epoch)
        train_pro[epoch - 1][0] = loss

            # 验证
        model.eval()
        val_loader = ValDataset(val_img_list, val_mask_list, opt.size)
        mdice, miou, mprecision, mrecall = val(val_loader, model)

        train_pro[epoch - 1][1] = mdice
        train_pro[epoch - 1][2] = miou
        train_pro[epoch - 1][3] = mprecision
        train_pro[epoch - 1][4] = mrecall

        print(f"epoch-{epoch}: mdice:{mdice:.4f}, miou:{miou:.4f}, "
              f"mprecision:{mprecision:.4f}, mrecall:{mrecall:.4f}")

        if mdice > best_dice:
            best_dice = mdice
            best_iou = miou
            best_precision = mprecision
            best_recall = mrecall
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_path, f'{opt.model}.pth'))
            print(f"★ New best at epoch {epoch}: dice={best_dice:.4f}")
            
        

    

