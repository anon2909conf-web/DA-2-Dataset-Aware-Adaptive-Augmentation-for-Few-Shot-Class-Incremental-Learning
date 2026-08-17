import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TinyImageNet200(Dataset):
    """
    200-class FSCIL dataset built from build_timgnet200.py (Tiny-ImageNet-200,
    re-split for FSCIL). Mirrors CUB200's exact session config as a COARSE
    counterpart -- 200 broad/diverse object categories (goldfish, school bus,
    teddy bear, sports car, ...) rather than one narrow super-category:
      - 100 base classes, many-shot
      - 10 incremental sessions of 10-way, 5 images/class each

    Directory layout expected under `root/timgnet200/`:
      images/<wnid>/<wnid>__<filename>
      split/train.csv, split/test.csv    (rows: filename,wnid -- filename already
                                           includes the wnid__ prefix written by
                                           build_timgnet200.py)
      index_list/session_1.txt ... session_10.txt
                                          (rows: "<wnid>/<wnid>__<filename>")
      class_order.txt                    (wnid <TAB> global_label, 0..199)

    Identical loading logic to Mix120/Mix200/Flowers102/Dogs120 -- only the
    root subfolder name, class count, and session count differ.
    """

    def __init__(self, root='./dataset', train=True, index_path=None, index=None,
                 base_sess=None, crop_transform=None, secondary_transform=None):
        self.root = os.path.expanduser(root)
        self.TIN_ROOT = osp.join(self.root, 'timgnet200')
        self.IMAGE_PATH = osp.join(self.TIN_ROOT, 'images')
        self.SPLIT_PATH = osp.join(self.TIN_ROOT, 'split')
        self.INDEX_PATH = osp.join(self.TIN_ROOT, 'index_list')

        self.crop_transform = crop_transform
        self.secondary_transform = secondary_transform
        if isinstance(secondary_transform, list):
            assert len(secondary_transform) == self.crop_transform.N_large + self.crop_transform.N_small
        self.multi_train = False

        # global wnid -> label map, fixed by build_timgnet200.py so it's identical
        # across every process/run (base session, every incremental session, and test)
        self.label_of = {}
        with open(osp.join(self.TIN_ROOT, 'class_order.txt'), 'r') as f:
            for line in f:
                wnid, lb = line.strip().split('\t')
                self.label_of[wnid] = int(lb)

        setname = 'train' if train else 'test'
        csv_path = osp.join(self.SPLIT_PATH, f'{setname}.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        for line in lines:
            fname, wnid = line.split(',')
            path = osp.join(self.IMAGE_PATH, wnid, fname)
            label = self.label_of[wnid]
            self.data.append(path)
            self.targets.append(label)
            self.data2label[path] = label
        self.targets = np.array(self.targets)

        if train:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(index_path)
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def SelectfromTxt(self, index_path):
        """index_path points at one of index_list/session_i.txt, whose lines are
        '<wnid>/<wnid>__<filename>' -- robust to nesting, unlike a positional split()."""
        with open(index_path, 'r') as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
        data_tmp, targets_tmp = [], []
        for rel in lines:
            img_path = osp.join(self.IMAGE_PATH, rel)
            data_tmp.append(img_path)
            targets_tmp.append(self.data2label[img_path])
        return data_tmp, np.array(targets_tmp)

    def SelectfromClasses(self, data, targets, index):
        data_tmp, targets_tmp = [], []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        return data_tmp, np.array(targets_tmp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        if self.multi_train:
            image = Image.open(path).convert('RGB')
            classify_image = [self.transform(image)]
            multi_crop, multi_crop_params = self.crop_transform(image)
            assert len(multi_crop) == self.crop_transform.N_large + self.crop_transform.N_small
            if isinstance(self.secondary_transform, list):
                multi_crop = [tf(x) for tf, x in zip(self.secondary_transform, multi_crop)]
            else:
                multi_crop = [self.secondary_transform(x) for x in multi_crop]
            total_image = classify_image + multi_crop
        else:
            total_image = self.transform(Image.open(path).convert('RGB'))
        return total_image, targets