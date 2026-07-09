# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:15:52 2026

@author: Administrator
"""

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import random
import torch

def split_with_val(img_dir, mask_dir, val_list_file):
    with open(val_list_file, 'r') as f:
        val_set = set(line.strip() for line in f)

    img_all = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))])

    train_imgs, train_masks = [], []
    val_imgs, val_masks = [], []

    for name in img_all:
        img_path = os.path.join(img_dir, name)
        mask_path = os.path.join(mask_dir, name)
        if name in val_set:
            val_imgs.append(img_path)
            val_masks.append(mask_path)
        else:
            train_imgs.append(img_path)
            train_masks.append(mask_path)

    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")
    return train_imgs, train_masks, val_imgs, val_masks


def clahe_enhance(img):
    import cv2
    import numpy as np
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_eq)


def random_flip(img, mask):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.3:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

def random_rotate(img, mask):
    if random.random() < 0.4:
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, expand=False)
        mask = mask.rotate(angle, expand=False)
    return img, mask

def random_scale(img, mask, target_size):
    if random.random() < 0.3:
        scale = random.uniform(0.85, 1.15)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        img = transforms.Resize((target_size, target_size))(img)
        mask = transforms.Resize((target_size, target_size))(mask)
    return img, mask


class AugTrainDataset(data.Dataset):
    def __init__(self, img_list, gt_list, trainsize):
        self.trainsize = trainsize
        self.images = sorted(img_list)
        self.gts = sorted(gt_list)
        self.filter_files()
        self.size = len(self)

        self.img_norm = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_norm = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize),
                             interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def filter_files(self):
        imgs, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path).convert("L")
            if img.size == gt.size:
                imgs.append(img_path)
                gts.append(gt_path)
        self.images, self.gts = imgs, gts

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.gts[index]).convert("L")

        if random.random() < 0.2:
            img = clahe_enhance(img)

        img, mask = random_scale(img, mask, self.trainsize)
        img, mask = random_rotate(img, mask)
        img, mask = random_flip(img, mask)

        img_tensor = self.img_norm(img)
        mask_tensor = self.gt_norm(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.images)

def get_loader(img_list, gt_list, batchsize, trainsize, seed=42):
    dataset = AugTrainDataset(img_list, gt_list, trainsize)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,         
        pin_memory=False,
        drop_last=True,
        generator=g
    )
    return loader


class ValDataset:
    def __init__(self, img_list, gt_list, testsize):
        self.imgs = img_list
        self.gts = gt_list
        self.testsize = testsize
        self.size = len(img_list)
        self.index = 0

        self.transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((testsize, testsize),
                             interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def load_data(self):
        img_path = self.imgs[self.index]
        gt_path = self.gts[self.index]
        name = img_path.split('/')[-1]
        if name.endswith('.jpg'):
            name = name.replace('.jpg', '.png')

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')

        img_tensor = self.transform(img).unsqueeze(0)
        gt_tensor = self.gt_transform(gt)

        self.index += 1
        return img_tensor, gt, name



class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
