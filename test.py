import torch
import os
import argparse
import torch.nn.functional as F
from Net import Net
from dataloader import test_dataset
import numpy as np
from PIL import Image
from val import test_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CVC-ClinicDB')
    parser.add_argument('--model', type=str, default='Net', help='model name')
    parser.add_argument('--pth_path', type=str, default='./snapshots/CVC-ClinicDB/Net/Net.pth')
    parser.add_argument('--trainsize', type=int, default=256)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--train_save', type=str, default='Net', help="Model save folder name")
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    model = Net().cuda()
    model.load_state_dict(torch.load(opt.pth_path))
 
    test_image_root = f'./data/{opt.data}/test/images/'
    test_gt_root = f'./data/{opt.data}/test/masks/'
    result_path = f'./results/{opt.data}/{opt.train_save}/'
    predict_path = f'./predicts/{opt.data}/{opt.train_save}/{opt.model}/'
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(predict_path, exist_ok=True)
    test_loader = test_dataset(test_image_root, test_gt_root, testsize=opt.trainsize)
    test_save(test_loader, model.eval(), predict_path, result_path, opt.data, opt.model)