import torch
import torch.nn as nn
import os
from PIL import Image
import cv2
import numpy as np
from collections import namedtuple

CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])
# CodeRow = namedtuple('CodeRow', ['top', 'middle', 'bottom', 'filename'])


class txtDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path, images_path, labels_path, transform=None, target_transform=None):
        super(txtDataset, self).__init__()
        fh = open(txt_path, 'r')
        imgs = []
        labels = []
        class_names = []
        for line in fh:
            line = line.rstrip()
            img_label = line.split(' ')
            imgs.append(img_label[0])
            labels.append(img_label[1])
            class_names.append(img_label[-1])

        self.imgs = imgs
        self.labels = labels
        self.class_names = class_names
        self.labels_path = []
        self.images_path = images_path
        self.labels_path = labels_path
        self.transoform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image_name = self.imgs[index]
        label_name = self.labels[index]
        class_name = self.class_names[index]
        image = []
        for item in image_name:
            img_path = os.path.join(self.images_path, item)
            # img = Image.open(img_path).convert('L')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image.append(img)
        image = np.array(image).transpose((1, 2, 0))

        label_path = os.path.join(self.labels_path, label_name)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = np.array(label).transpose((0, 1))
        image = Image.fromarray(image)
        label = Image.fromarray(label)

        if self.transoform is not None:
            image = self.transoform(image)
            label = self.transoform(label)
        else:
            image = torch.Tensor(image)
            label = torch.Tensor(label)

        return image, label, label_path, class_name
        # return image, label, label_name

    def __len__(self):
        return len(self.imgs)

class ImageFileDataset(txtDataset):
    def __getitem__(self, index):
        sample, target, target_path = super().__getitem__(index)

        return sample, target, target_path

if  __name__ == '__main__':
    from torch.utils.data import DataLoader

    txt_path = './data/pair.txt'
    images_path = './data'
    labels_path = './data'

    dataset = txtDataset(txt_path, images_path, labels_path)

    loader = DataLoader(
        dataset, batch_size=1, num_workers=2
    )
    for i, (img, label, _, _) in enumerate(loader):
        print(img.size())
        print(label.size())

