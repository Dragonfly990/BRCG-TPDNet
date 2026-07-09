import numpy as np
import imageio
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

epsilon = 1e-7

def Dice(y_pred, y_true):
    y_pred = y_pred.astype(np.float32).flatten()
    y_true = y_true.astype(np.float32).flatten()
    intersection = np.sum(y_pred * y_true)
    return (2.0 * intersection + epsilon) / (np.sum(y_pred) + np.sum(y_true) + epsilon)

def IoU(y_pred, y_true):
    y_pred = y_pred.astype(np.float32).flatten()
    y_true = y_true.astype(np.float32).flatten()
    intersection = np.sum(y_pred * y_true)
    union = np.sum(y_pred) + np.sum(y_true) - intersection
    return (intersection + epsilon) / (union + epsilon)

def compulate_metrics(y_true, y_pred):
    y_pred_bin = (y_pred > 0.5).astype(np.float32)
    y_true = y_true.astype(np.float32)
    tp = np.sum(y_true * y_pred_bin)
    tn = np.sum((1 - y_true) * (1 - y_pred_bin))
    fp = np.sum((1 - y_true) * y_pred_bin)
    fn = np.sum(y_true * (1 - y_pred_bin))
    rec = tp / (tp + fn + epsilon)
    prec = tp / (tp + fp + epsilon)
    return rec, prec


def val(val_loader, model):
    model.eval()
    dice_scores, iou_scores = [], []
    precision_scores, recall_scores = [], []
    
    thresh = 0.5
    for i in range(val_loader.size):
        image, gt, name = val_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt = (gt > 128).astype(np.float32)

        image = image.cuda()

        with torch.no_grad():
            _,_,map_2 = model(image)
        res =map_2   # logits

        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=True)
        res = res.sigmoid().data.cpu().numpy().squeeze()

        pred_bin = (res > thresh).astype(np.float32)

        dice_scores.append(Dice(pred_bin, gt))
        iou_scores.append(IoU(pred_bin, gt))
        rec, prec = compulate_metrics(gt, res)
        recall_scores.append(rec)
        precision_scores.append(prec)
       
    return (np.mean(dice_scores), np.mean(iou_scores),
            np.mean(precision_scores), np.mean(recall_scores))

def test_save(test_loader, model, predict_path, result_path, data_name, model_name):
    model.eval()
    os.makedirs(predict_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    dice_scores, iou_scores = [], []
    precision_scores, recall_scores = [], []
   
    thresh = 0.5
    for i in tqdm(range(test_loader.size), total=test_loader.size):
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt = (gt > 128).astype(np.float32)

        image = image.cuda()

       
        with torch.no_grad():
            _,_,map_2 = model(image)
        res = map_2
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=True)
        res = res.sigmoid().data.cpu().numpy().squeeze()

        res_save = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        imageio.imwrite(predict_path + name, res_save)

        pred_bin = (res > thresh).astype(np.float32)

        dice_scores.append(Dice(pred_bin, gt))
        iou_scores.append(IoU(pred_bin, gt))
        rec, prec = compulate_metrics(gt, res)
        recall_scores.append(rec)
        precision_scores.append(prec)
        

    mdice = np.mean(dice_scores)
    miou = np.mean(iou_scores)
    mprecision = np.mean(precision_scores)
    mrecall = np.mean(recall_scores)
   

    print(f'{data_name}_final: dice={mdice:.4f}, iou={miou:.4f},precision={mprecision:.4f}, recall={mrecall:.4f}')

    result = np.array([mdice, miou, mprecision, mrecall])
    np.savetxt(os.path.join(result_path, f'{model_name}.txt'), result)

    

