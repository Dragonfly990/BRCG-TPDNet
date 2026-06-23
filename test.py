import torch
import os
import argparse
import torch.nn.functional as F
from lib.Net import Net
from utils.dataloader import test_loader
import numpy as np
from PIL import Image

def calculate_metrics(pred, gt):
    """
    pred/gt: torch tensor 0/1 binary mask
    return dice, iou, precision, recall, acc, specificity
    """
    pred = pred.float()
    gt = gt.float()

    # Dice & IoU
    inter = (pred * gt).sum()
    sum_pred = pred.sum()
    sum_gt = gt.sum()
    dice = (2 * inter + 1e-6) / (sum_pred + sum_gt + 1e-6)
    iou = (inter + 1e-6) / (sum_pred + sum_gt - inter + 1e-6)

    # TP TN FP FN
    tp = (pred * gt).sum()
    tn = ((1 - pred) * (1 - gt)).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()

    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    

    return (
        dice.item(),
        iou.item(),
        precision.item(),
        recall.item()
    )

def test(model, test_loader, save_pred=False, save_dir="./pred_result"):
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
   

    sample_num = len(test_loader)
    if save_pred:
        os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (image, gt, name) in enumerate(test_loader):
            image, gt = image.cuda(), gt.cuda()
            _, _, pred_out = model(image)
            pred_out = torch.sigmoid(pred_out)
            pred_out = F.interpolate(pred_out, size=gt.shape[2:], mode="bilinear", align_corners=False)
            pred_bin = (pred_out > 0.5).float()

            dice, iou, prec, rec, acc, spec = calculate_metrics(pred_bin, gt)
            total_dice += dice
            total_iou += iou
            total_precision += prec
            total_recall += rec
            

            if save_pred:
                pred_np = pred_bin.squeeze().cpu().numpy() * 255
                pred_img = Image.fromarray(pred_np.astype(np.uint8))
                pred_img.save(os.path.join(save_dir, name[0]))
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx+1}/{sample_num}")

    avg_dice = total_dice / sample_num
    avg_iou = total_iou / sample_num
    avg_precision = total_precision / sample_num
    avg_recall = total_recall / sample_num
    

    print("\n==== Test Metric Result ====")
    print(f"Average Dice:      {avg_dice:.4f}")
    print(f"Average IoU:       {avg_iou:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:    {avg_recall:.4f}")
   
    return avg_dice, avg_iou, avg_precision, avg_recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CVC_ClinicDB')
    parser.add_argument('--pth_path', type=str, default='./snapshots/Net_Seg/Net-best.pth')
    parser.add_argument('--trainsize', type=int, default=256)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_pred', action='store_true')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    model = Net().cuda()
    model.load_state_dict(torch.load(opt.pth_path))
    print(f"Load weights: {opt.pth_path}")
    test_img_root = f'./data/{opt.data}/test/images/'
    test_gt_root = f'./data/{opt.data}/test/masks/'
    test_loader = test_loader(test_img_root, test_gt_root, batchsize=1, trainsize=opt.trainsize)
    test(model, test_loader, save_pred=opt.save_pred)
