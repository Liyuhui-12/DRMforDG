# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import h5py
import os
import random
import copy
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset,Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

from os import path as osp
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    "CelebA",
    'DSprites',
    'dshapes'
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)




class dshapes(MultipleDomainDataset):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    CHECKPOINT_FREQ = 300

    def __init__(self, root, test_envs, hparams):

        self.input_shape = (3, 28, 28,)
        self.num_classes = 2

        self.image_dir = osp.join(root, '3dshapes/3dshapes.h5')
        self.attr_path = osp.join(root, '3dshapes/list_attr_3dshapes.txt')
        self.domain = 'floor_hue'
        self.label = 'orientation'

        print('domain', self.domain, 'label', self.label)

        self.environments = [0.1, 0.2, 0.9]
        # self.environments = [0.5, 0.5, 0.5]
        self.label_noise = 0.25

        self.environments = [1 - i for i in self.environments]

        self.num_domain = len(self.environments)
        self.selected_attrs = [self.domain, self.label]
        self.attr2idx = {}
        self.idx2attr = {}
        self.ordataset = [[] for _ in range(self.num_domain)]

        self.imgpaths = []
        self.domains = []
        self.labels = []

        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        lines = lines[2:]
        # random.seed(1234)
        random.shuffle(lines)

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            attrlabel = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                attrlabel.append(values[idx] == '1')
            self.labels.append(attrlabel[-1])
            self.domains.append(attrlabel[0])
            self.imgpaths.append(filename)

        labels = torch.tensor(self.labels).float()
        domains = torch.tensor(self.domains).float()

        maxs = len(labels)
        # print(labels)
        # print(domains)
        for i in range(2):
            for j in range(2):
                p = 0.5 * 0.5
                tmpnum = ((labels == i) & (domains == j)).sum().item() / p
                if tmpnum < maxs:
                    maxs = tmpnum

        maxs = int(maxs)
        maxs = min(100000, maxs)

        cusum0 = {}
        for i in range(2):
            for j in range(2):
                cusum0.update({(i, j): 0})

        self.labels = []
        self.domains = []
        self.imgpaths = []
        print("maxs:", maxs)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            attrlabel = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                attrlabel.append(values[idx] == '1')

            label = attrlabel[-1]
            domain = attrlabel[0]
            if cusum0[(label, domain)] < maxs * 0.5 * 0.5:
                cusum0[(label, domain)] += 1
                self.labels.append(label)
                self.domains.append(domain)
                self.imgpaths.append(filename)

        labels = torch.tensor(self.labels).float()
        orlabels = labels.clone()

        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(self.label_noise, len(labels)))

        self.labels = labels.tolist()

        domains = torch.tensor(self.domains).float()

        equalp = sum(self.environments) / len(self.environments)

        maxs = len(labels)

        for i in range(2):
            for j in range(2):
                if i == j:
                    p = equalp * 0.5
                else:
                    p = (1 - equalp) * 0.5
                tmpnum = ((labels == i) & (domains == j)).sum().item() / p
                if tmpnum < maxs:
                    maxs = tmpnum

        maxs = int(maxs)


        fenvironments = [1 - i for i in self.environments]

        maxsnum = {}
        cusum = {}
        for i in range(2):
            for j in range(2):
                for k in range(self.num_domain):
                    if i == j:
                        p = equalp * 0.5 * self.environments[k] / sum(self.environments)
                    else:
                        p = (1 - equalp) * 0.5 * fenvironments[k] / sum(fenvironments)
                    maxsnum.update({(i, j, k): maxs * p})
                    cusum.update({(i, j, k): 0})

        # print(self.imgpaths)
        # print(self.labels)
        # print(self.domains)
        # print(orlabels)
        for i, (filename, label, domain, orlabels) in enumerate(
                zip(self.imgpaths, self.labels, self.domains, orlabels)):

            l = label
            d = domain
            klist = list(range(self.num_domain))
            random.shuffle(klist)
            for k in klist:
                if cusum[(l, d, k)] < maxsnum[(l, d, k)]:
                    cusum[(l, d, k)] += 1
                    self.ordataset[k].append([filename, label, domain, orlabels])
                    break

        self.datasets = []

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])

        # print(self.ordataset)
        all_imgs = h5py.File(self.image_dir, 'r')['images']
        all_imgs = all_imgs[:480000]
        for id, i in enumerate(self.ordataset):
            imgs = []
            ls = []
            ds = []
            ols = []
            for filename, label, domain, orlabels in i:
                ls.append(label)
                ds.append(domain)
                ols.append(orlabels)
                image = Image.fromarray(all_imgs[int(filename)])
                image = transform(image)
                imgs.append(image)
            imgs = torch.stack(imgs)
            ls = torch.tensor(ls)
            ds = torch.tensor(ds)
            ols = torch.tensor(ols)

            # shuffle = torch.randperm(len(ls))
            # ls=ls[shuffle]

            ls = ls.long()
            ols = ols.long()
            ds = ds.long()



            self.datasets.append(TensorDataset(imgs, ls))

    def read_image(self, img_path):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'.".format(img_path))
                pass
        return img

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class DSprites(MultipleDomainDataset):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    CHECKPOINT_FREQ = 300

    def __init__(self, root, test_envs, hparams):

        self.input_shape = (1, 28, 28,)
        self.num_classes = 2

        self.image_dir = osp.join(root, 'DSprites/Img/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        self.attr_path = osp.join(root, 'DSprites/Anno/list_attr_dsprites.txt')
        self.domain = 'PosX'
        self.label = 'PosY'

        print('domain', self.domain, 'label', self.label)

        self.environments = [0.1, 0.2, 0.9]
        # self.environments = [0.5, 0.5, 0.5]
        self.label_noise = 0.25

        self.environments = [1 - i for i in self.environments]

        self.num_domain = len(self.environments)
        self.selected_attrs = [self.domain, self.label]
        self.attr2idx = {}
        self.idx2attr = {}
        self.ordataset = [[] for _ in range(self.num_domain)]

        self.imgpaths = []
        self.domains = []
        self.labels = []

        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        lines = lines[2:]
        # random.seed(1234)
        random.shuffle(lines)

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            attrlabel = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                attrlabel.append(values[idx] == '1')
            self.labels.append(attrlabel[-1])
            self.domains.append(attrlabel[0])
            self.imgpaths.append(filename)

        labels = torch.tensor(self.labels).float()
        domains = torch.tensor(self.domains).float()

        maxs = len(labels)



        for i in range(2):
            for j in range(2):
                p = 0.5 * 0.5
                tmpnum = ((labels == i) & (domains == j)).sum().item() / p
                if tmpnum < maxs:
                    maxs = tmpnum

        maxs = int(maxs)
        maxs = min(100000, maxs)

        cusum0 = {}
        for i in range(2):
            for j in range(2):
                cusum0.update({(i, j): 0})

        self.labels = []
        self.domains = []
        self.imgpaths = []
        print("maxs:", maxs)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            attrlabel = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                attrlabel.append(values[idx] == '1')

            label = attrlabel[-1]
            domain = attrlabel[0]
            if cusum0[(label, domain)] < maxs * 0.5 * 0.5:
                cusum0[(label, domain)] += 1
                self.labels.append(label)
                self.domains.append(domain)
                self.imgpaths.append(filename)

        labels = torch.tensor(self.labels).float()
        orlabels = labels.clone()

        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(self.label_noise, len(labels)))

        self.labels = labels.tolist()

        domains = torch.tensor(self.domains).float()

        equalp = sum(self.environments) / len(self.environments)

        maxs = len(labels)

        for i in range(2):
            for j in range(2):
                if i == j:
                    p = equalp * 0.5
                else:
                    p = (1 - equalp) * 0.5
                tmpnum = ((labels == i) & (domains == j)).sum().item() / p
                if tmpnum < maxs:
                    maxs = tmpnum

        maxs = int(maxs)


        fenvironments = [1 - i for i in self.environments]

        maxsnum = {}
        cusum = {}
        for i in range(2):
            for j in range(2):
                for k in range(self.num_domain):
                    if i == j:
                        p = equalp * 0.5 * self.environments[k] / sum(self.environments)
                    else:
                        p = (1 - equalp) * 0.5 * fenvironments[k] / sum(fenvironments)
                    maxsnum.update({(i, j, k): maxs * p})
                    cusum.update({(i, j, k): 0})


        for i, (filename, label, domain, orlabels) in enumerate(
                zip(self.imgpaths, self.labels, self.domains, orlabels)):

            l = label
            d = domain
            klist = list(range(self.num_domain))
            random.shuffle(klist)
            for k in klist:
                if cusum[(l, d, k)] < maxsnum[(l, d, k)]:
                    cusum[(l, d, k)] += 1
                    self.ordataset[k].append([filename, label, domain, orlabels])
                    break

        self.datasets = []

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])

        # print(self.ordataset)
        all_imgs = np.load(self.image_dir, allow_pickle=True)["imgs"]
        for id, i in enumerate(self.ordataset):
            imgs = []
            ls = []
            ds = []
            ols = []
            for filename, label, domain, orlabels in i:
                ls.append(label)
                ds.append(domain)
                ols.append(orlabels)
                image = Image.fromarray(all_imgs[int(filename)] * 255, "L")

                image = transform(image)
                imgs.append(image)
            imgs = torch.stack(imgs)
            ls = torch.tensor(ls)
            ds = torch.tensor(ds)
            ols = torch.tensor(ols)

            # shuffle = torch.randperm(len(ls))
            # ls=ls[shuffle]

            ls = ls.long()
            ols = ols.long()
            ds = ds.long()



            self.datasets.append(TensorDataset(imgs, ls))

    def read_image(self, img_path):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'.".format(img_path))
                pass
        return img

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()




class TensorDataset_b_trans(Dataset):
    def __init__(self, *tensors,transform=None):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.transform=transform
        self.tensors = tensors


    def __getitemnb__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)



    def read_image(self,img_path):

        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'.".format(img_path))
                pass
        return img

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path,target = self.__getitemnb__(index)
        sample = self.read_image(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return self.tensors[1].size(0)

class CelebA(MultipleDomainDataset):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    CHECKPOINT_FREQ = 300

    def __init__(self, root, test_envs, hparams):

        self.input_shape = (3, 224, 224,)
        self.num_classes = 2

        self.image_dir = osp.join(root,'CelebA/Img/img_align_celeba')
        self.attr_path = osp.join(root,'CelebA/Anno/list_attr_celeba.txt')

        self.domain = 'Wearing_Hat'
        self.label = 'No_Beard'

        print('domain', self.domain, 'label', self.label)

        self.environments = [0.1, 0.2, 0.9]
        # self.environments = [0.5, 0.5, 0.5]
        self.label_noise = 0.25

        self.environments = [1 - i for i in self.environments]

        self.num_domain = len(self.environments)
        self.selected_attrs = [self.domain, self.label]
        self.attr2idx = {}
        self.idx2attr = {}
        self.ordataset = [[] for _ in range(self.num_domain)]

        self.imgpaths = []
        self.domains = []
        self.labels = []

        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        lines = lines[2:]
        # random.seed(1234)
        random.shuffle(lines)

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            attrlabel = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                attrlabel.append(values[idx] == '1')
            self.labels.append(attrlabel[-1])
            self.domains.append(attrlabel[0])
            self.imgpaths.append(filename)


        labels = torch.tensor(self.labels).float()
        domains = torch.tensor(self.domains).float()

        maxs = len(labels)

        for i in range(2):
            for j in range(2):
                p=0.5*0.5
                tmpnum = ((labels == i) & (domains == j)).sum().item() / p
                if tmpnum < maxs:
                    maxs = tmpnum

        maxs = int(maxs)


        cusum0 = {}
        for i in range(2):
            for j in range(2):
                cusum0.update({(i, j): 0})



        self.labels=[]
        self.domains=[]
        self.imgpaths=[]

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            attrlabel = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                attrlabel.append(values[idx] == '1')

            label=attrlabel[-1]
            domain=attrlabel[0]
            if cusum0[(label,domain)]<maxs*0.5*0.5:
                cusum0[(label, domain)]+=1
                self.labels.append(label)
                self.domains.append(domain)
                self.imgpaths.append(filename)





        labels = torch.tensor(self.labels).float()
        orlabels=labels.clone()

        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(self.label_noise, len(labels)))



        self.labels=labels.tolist()

        domains = torch.tensor(self.domains).float()

        equalp = sum(self.environments) / len(self.environments)

        maxs = len(labels)

        for i in range(2):
            for j in range(2):
                if i == j:
                    p = equalp * 0.5
                else:
                    p = (1 - equalp) * 0.5
                tmpnum = ((labels == i) & (domains == j)).sum().item() / p
                if tmpnum < maxs:
                    maxs = tmpnum

        maxs = int(maxs)

        fenvironments = [1 - i for i in self.environments]

        maxsnum = {}
        cusum = {}
        for i in range(2):
            for j in range(2):
                for k in range(self.num_domain):
                    if i == j:
                        p = equalp * 0.5 * self.environments[k] / sum(self.environments)
                    else:
                        p = (1 - equalp) * 0.5 * fenvironments[k] / sum(fenvironments)
                    maxsnum.update({(i, j, k): maxs * p})
                    cusum.update({(i, j, k): 0})





        for i, (filename, label, domain,orlabels) in enumerate(zip(self.imgpaths,self.labels,self.domains,orlabels)):

            l=label
            d=domain
            klist=list(range(self.num_domain))
            random.shuffle(klist)
            for k in klist:
                if cusum[(l, d, k)] < maxsnum[(l, d, k)]:
                    cusum[(l, d, k)] += 1
                    self.ordataset[k].append([filename, label, domain,orlabels])
                    break


        self.datasets = []

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for id,i in enumerate(self.ordataset):
            imgs = []
            ls = []
            ds = []
            ols=[]
            for filename, label, domain,orlabels in i:
                ls.append(label)
                ds.append(domain)
                ols.append(orlabels)
                image = self.read_image(os.path.join(self.image_dir, filename))
                image = transform(image)
                imgs.append(image)
            imgs = torch.stack(imgs)
            ls = torch.tensor(ls)
            ds = torch.tensor(ds)
            ols=torch.tensor(ols)

            # shuffle = torch.randperm(len(ls))
            # ls=ls[shuffle]


            ls = ls.long()
            ols = ols.long()
            ds=ds.long()



            self.datasets.append(TensorDataset(imgs, ls))

    def read_image(self, img_path):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'.".format(img_path))
                pass
        return img

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class CelebA_1(MultipleDomainDataset):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    CHECKPOINT_FREQ = 300

    def __init__(self, root, test_envs, hparams):

        self.input_shape = (3, 224, 224,)
        self.num_classes = 2

        self.image_dir = osp.join(root,'CelebA/Img/img_align_celeba')
        self.attr_path = osp.join(root,'CelebA/Anno/list_attr_celeba.txt')



        self.domain = 'Wearing_Necktie'
        self.label = 'Smiling'


        print('domain', self.domain, 'label', self.label)

        self.environments = [0.1, 0.2, 0.9]
        # self.environments = [0.5, 0.5, 0.5]
        self.label_noise = 0.25

        self.environments = [1 - i for i in self.environments]

        self.num_domain = len(self.environments)
        self.selected_attrs = [self.domain, self.label]
        self.attr2idx = {}
        self.idx2attr = {}
        self.ordataset = [[] for _ in range(self.num_domain)]

        self.imgpaths = []
        self.domains = []
        self.labels = []

        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        lines = lines[2:]
        # random.seed(1234)
        random.shuffle(lines)

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            attrlabel = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                attrlabel.append(values[idx] == '1')
            self.labels.append(attrlabel[-1])
            self.domains.append(attrlabel[0])
            self.imgpaths.append(filename)


        labels = torch.tensor(self.labels).float()
        domains = torch.tensor(self.domains).float()

        maxs = len(labels)

        for i in range(2):
            for j in range(2):
                p=0.5*0.5
                tmpnum = ((labels == i) & (domains == j)).sum().item() / p
                if tmpnum < maxs:
                    maxs = tmpnum

        maxs = int(maxs)


        cusum0 = {}
        for i in range(2):
            for j in range(2):
                cusum0.update({(i, j): 0})



        self.labels=[]
        self.domains=[]
        self.imgpaths=[]

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            attrlabel = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                attrlabel.append(values[idx] == '1')

            label=attrlabel[-1]
            domain=attrlabel[0]
            if cusum0[(label,domain)]<maxs*0.5*0.5:
                cusum0[(label, domain)]+=1
                self.labels.append(label)
                self.domains.append(domain)
                self.imgpaths.append(filename)





        labels = torch.tensor(self.labels).float()
        orlabels=labels.clone()

        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(self.label_noise, len(labels)))



        self.labels=labels.tolist()

        domains = torch.tensor(self.domains).float()

        equalp = sum(self.environments) / len(self.environments)

        maxs = len(labels)

        for i in range(2):
            for j in range(2):
                if i == j:
                    p = equalp * 0.5
                else:
                    p = (1 - equalp) * 0.5
                tmpnum = ((labels == i) & (domains == j)).sum().item() / p
                if tmpnum < maxs:
                    maxs = tmpnum

        maxs = int(maxs)

        fenvironments = [1 - i for i in self.environments]

        maxsnum = {}
        cusum = {}
        for i in range(2):
            for j in range(2):
                for k in range(self.num_domain):
                    if i == j:
                        p = equalp * 0.5 * self.environments[k] / sum(self.environments)
                    else:
                        p = (1 - equalp) * 0.5 * fenvironments[k] / sum(fenvironments)
                    maxsnum.update({(i, j, k): maxs * p})
                    cusum.update({(i, j, k): 0})





        for i, (filename, label, domain,orlabels) in enumerate(zip(self.imgpaths,self.labels,self.domains,orlabels)):

            l=label
            d=domain
            klist=list(range(self.num_domain))
            random.shuffle(klist)
            for k in klist:
                if cusum[(l, d, k)] < maxsnum[(l, d, k)]:
                    cusum[(l, d, k)] += 1
                    self.ordataset[k].append([filename, label, domain,orlabels])
                    break


        self.datasets = []

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for id,i in enumerate(self.ordataset):
            imgs = []
            ls = []
            ds = []
            ols=[]
            for filename, label, domain,orlabels in i:
                ls.append(label)
                ds.append(domain)
                ols.append(orlabels)
                image = self.read_image(os.path.join(self.image_dir, filename))
                image = transform(image)
                imgs.append(image)
            imgs = torch.stack(imgs)
            ls = torch.tensor(ls)
            ds = torch.tensor(ds)
            ols=torch.tensor(ols)

            # shuffle = torch.randperm(len(ls))
            # ls=ls[shuffle]


            ls = ls.long()
            ols = ols.long()
            ds=ds.long()



            self.datasets.append(TensorDataset(imgs, ls))

    def read_image(self, img_path):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'.".format(img_path))
                pass
        return img

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()



class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)

