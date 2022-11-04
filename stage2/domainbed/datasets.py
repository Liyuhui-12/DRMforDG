# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import time
import h5py
import os
import torch
import numpy as np
import copy
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset,Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

import random
from os import path as osp

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

from domainbed import networks

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
    "CelebA_1",
    "CelebA0",

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


class dshapes(MultipleDomainDataset):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    CHECKPOINT_FREQ = 300

    def __init__(self, root, test_envs, hparams, k1, k2, fenv, data):

        self.input_shape = data.input_shape
        self.num_classes = data.num_classes

        del data.datasets

        self.network = networks.clsdd(data.input_shape, hparams, data.num_classes, data.num_domain - 1)

        self.network.cuda()
        state_dict = torch.load('./weights/model_dshapes_target_{}.ckpt'.format(fenv))
        self.network.load_state_dict(state_dict['model'], strict=True)

        self.network.eval()

        self.bs = 512
        self.k1 = k1
        self.k2 = k2
        self.envid = 0

        self.ordataset = data.ordataset

        self.datasets = []

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])

        all_imgs = h5py.File(data.image_dir, 'r')['images']
        all_imgs = all_imgs[:480000]
        for id, i in enumerate(self.ordataset):

            key1 = self.k1[id]
            key2 = self.k2[id]

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

            #if not osp.exists(os.path.join('saveind', 'index_{}_t{}_{}_{}_done'.format('DSprites', fenv, ts, id))):
            loadst=time.time()
            xid = 0
            z = []
            with torch.no_grad():
                while xid < imgs.shape[0]:
                    tmpx = imgs[xid:xid + self.bs]
                    tmpx = tmpx.cuda()
                    tmpp = self.network(tmpx)

                    z.extend(tmpp.tolist())

                    xid += self.bs

                z = torch.tensor(z)

                n = z.shape[0]

                keybool = [0 if i in key1 else 1 for i in range(n)]
                keybool = torch.tensor(keybool)
                print('load time:', time.time() - loadst)
                print('n', n)
                indst = time.time()


                if n<20000:

                    distmat = torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                              torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n).t()
                    distmat.addmm_(beta=1, alpha=-2, mat1=z, mat2=z.t())

                    ye = ls.expand(n, n)
                    ke = keybool.expand(n, n)
                    k_notequal = (ke != ke.t())
                    y_notequal = (ye != ye.t())




                    distmat = distmat * y_notequal + distmat.max() * 2 * (~y_notequal) + distmat.max() * 2 * k_notequal
                    #distmat = distmat * y_notequal + 999 * 2 * (~y_notequal) + 999 * 2 * k_notequal
                    index = distmat.min(0).indices

                else:
                    index_list=[]
                    z=z.cuda()
                    ls=ls.cuda()
                    keybool=keybool.cuda()


                    for i_s in range(n):
                        tmpz = z[i_s].unsqueeze(0)
                        tmpdistmat = torch.pow(tmpz, 2).sum(dim=1, keepdim=True).expand(1, n) + \
                                     torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, 1).t()
                        tmpdistmat.addmm_(beta=1, alpha=-2, mat1=tmpz, mat2=z.t())
                        tmpy_notequal = (ls != ls[i_s])
                        tmpk_notequal = (keybool != keybool[i_s])

                        tmpdistmat = tmpdistmat * tmpy_notequal + tmpdistmat.max() * 2 * (~tmpy_notequal) + tmpdistmat.max() * 2 * tmpk_notequal
                        # tmpdistmat = tmpdistmat * tmpy_notequal + 999 * 2 * (
                        #      ~tmpy_notequal) + 999 * 2 * tmpk_notequal
                        index_list.append((tmpdistmat.min(1).indices).item())




                            # (tmpdistmat[0]-distmat[i_s]).abs().max()

                    index=torch.tensor(index_list)
                print('ind time',time.time()-indst)

                index = index.cpu()
                keybool = keybool.cpu()
                ls=ls.cpu()

                weights=[]
                for i_label in range(data.num_classes):
                    i_label_num = (ls == i_label).float().mean()
                    weights.append(i_label_num)
            # from IPython import embed
            # embed()
            # if not osp.exists('saveind'):
            #     os.makedirs('saveind')
            #
            # save_dict = {
            #     "index": index,
            #     "weights": weights,
            #     'keybool': keybool,
            # }
            # # from IPython import embed
            # # embed()
            # torch.save(save_dict, osp.join('saveind', 'index_{}_t{}_{}_{}'.format('DSprites', fenv, ts, id)))
            # with open(os.path.join('saveind', 'index_{}_t{}_{}_{}_done'.format('DSprites', fenv, ts, id)),
            #           'w') as f:
            #     f.write('index_{}_t{}_{}_{}_done'.format('DSprites', fenv, ts, id))
            index = index.cpu()

            # else:
            #     loaddict = torch.load(osp.join('saveind', 'index_{}_t{}_{}_{}'.format('DSprites', fenv, ts, id)))
            #     index = loaddict['index']
            #     weights = loaddict['weights']
            #     keybool = loaddict['keybool']






            ols = ols.long()
            ls = ls.long()
            ds = ds.long()













            # if id in test_envs:
            #     t=augment_transform
            # else:
            #     t=transform
            # self.datasets.append(TensorDataset_b_trans(imgs, ls, matchindex=index, weights=weights, keybool=keybool,
            #                                      inputshape=data.input_shape,transform=t))
            self.datasets.append(TensorDataset_b(imgs, ls, matchindex=index, weights=weights, keybool=keybool,
                                                 inputshape=data.input_shape))

        rs = []
        for i in self.datasets:
            rs.append((i.weights * i.n).sum())
        rsmin = np.array(rs).min()
        for id, i in enumerate(self.datasets):
            i.adjust_n(rsmin / rs[id])
            # from IPython import embed
            # embed()

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

    def __init__(self, root, test_envs, hparams, k1, k2, fenv, data):

        self.input_shape = data.input_shape
        self.num_classes = data.num_classes

        del data.datasets

        self.network = networks.clsdd(data.input_shape, hparams, data.num_classes, data.num_domain - 1)

        self.network.cuda()
        state_dict = torch.load('./weights/model_DSprites_target_{}.ckpt'.format(fenv))
        self.network.load_state_dict(state_dict['model'], strict=True)
        print('load!')

        self.network.eval()

        self.bs = 512
        self.k1 = k1
        self.k2 = k2
        self.envid = 0

        self.ordataset = data.ordataset

        self.datasets = []

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])

        all_imgs = np.load(data.image_dir, allow_pickle=True)["imgs"]
        for id, i in enumerate(self.ordataset):

            key1 = self.k1[id]
            key2 = self.k2[id]

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

            #if not osp.exists(os.path.join('saveind', 'index_{}_t{}_{}_{}_done'.format('DSprites', fenv, ts, id))):
            loadst=time.time()
            xid = 0
            z = []
            with torch.no_grad():
                while xid < imgs.shape[0]:
                    tmpx = imgs[xid:xid + self.bs]
                    tmpx = tmpx.cuda()
                    tmpp = self.network(tmpx)

                    z.extend(tmpp.tolist())

                    xid += self.bs

                z = torch.tensor(z)

                n = z.shape[0]

                keybool = [0 if i in key1 else 1 for i in range(n)]
                keybool = torch.tensor(keybool)
                print('load time:', time.time() - loadst)
                print('n', n)
                indst = time.time()


                if n<20000:

                    distmat = torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                              torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n).t()
                    distmat.addmm_(beta=1, alpha=-2, mat1=z, mat2=z.t())

                    ye = ls.expand(n, n)
                    ke = keybool.expand(n, n)
                    k_notequal = (ke != ke.t())
                    y_notequal = (ye != ye.t())



                    distmat = distmat * y_notequal + distmat.max() * 2 * (~y_notequal) + distmat.max() * 2 * k_notequal
                    #distmat = distmat * y_notequal + 999 * 2 * (~y_notequal) + 999 * 2 * k_notequal
                    index = distmat.min(0).indices

                else:
                    index_list=[]
                    z=z.cuda()
                    ls=ls.cuda()
                    keybool=keybool.cuda()


                    for i_s in range(n):
                        tmpz = z[i_s].unsqueeze(0)
                        tmpdistmat = torch.pow(tmpz, 2).sum(dim=1, keepdim=True).expand(1, n) + \
                                     torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, 1).t()
                        tmpdistmat.addmm_(beta=1, alpha=-2, mat1=tmpz, mat2=z.t())
                        tmpy_notequal = (ls != ls[i_s])
                        tmpk_notequal = (keybool != keybool[i_s])

                        tmpdistmat = tmpdistmat * tmpy_notequal + tmpdistmat.max() * 2 * (~tmpy_notequal) + tmpdistmat.max() * 2 * tmpk_notequal
                        # tmpdistmat = tmpdistmat * tmpy_notequal + 999 * 2 * (
                        #      ~tmpy_notequal) + 999 * 2 * tmpk_notequal
                        index_list.append((tmpdistmat.min(1).indices).item())




                            # (tmpdistmat[0]-distmat[i_s]).abs().max()

                    index=torch.tensor(index_list)
                print('ind time',time.time()-indst)

                index = index.cpu()
                keybool = keybool.cpu()
                ls=ls.cpu()

                weights=[]
                for i_label in range(data.num_classes):
                    i_label_num = (ls == i_label).float().mean()
                    weights.append(i_label_num)
            # from IPython import embed
            # embed()
            # if not osp.exists('saveind'):
            #     os.makedirs('saveind')
            #
            # save_dict = {
            #     "index": index,
            #     "weights": weights,
            #     'keybool': keybool,
            # }
            # # from IPython import embed
            # # embed()
            # torch.save(save_dict, osp.join('saveind', 'index_{}_t{}_{}_{}'.format('DSprites', fenv, ts, id)))
            # with open(os.path.join('saveind', 'index_{}_t{}_{}_{}_done'.format('DSprites', fenv, ts, id)),
            #           'w') as f:
            #     f.write('index_{}_t{}_{}_{}_done'.format('DSprites', fenv, ts, id))
            index = index.cpu()

            # else:
            #     loaddict = torch.load(osp.join('saveind', 'index_{}_t{}_{}_{}'.format('DSprites', fenv, ts, id)))
            #     index = loaddict['index']
            #     weights = loaddict['weights']
            #     keybool = loaddict['keybool']





            ols = ols.long()
            ls = ls.long()
            ds = ds.long()











            # if id in test_envs:
            #     t=augment_transform
            # else:
            #     t=transform
            # self.datasets.append(TensorDataset_b_trans(imgs, ls, matchindex=index, weights=weights, keybool=keybool,
            #                                      inputshape=data.input_shape,transform=t))
            self.datasets.append(TensorDataset_b(imgs, ls, matchindex=index, weights=weights, keybool=keybool,
                                                 inputshape=data.input_shape))

        rs = []
        for i in self.datasets:
            rs.append((i.weights * i.n).sum())
        rsmin = np.array(rs).min()
        for id, i in enumerate(self.datasets):
            i.adjust_n(rsmin / rs[id])
            # from IPython import embed
            # embed()

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




class CelebA(MultipleDomainDataset):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    CHECKPOINT_FREQ=300
    def __init__(self, root, test_envs, hparams,k1, k2,fenv,data):

        self.input_shape=data.input_shape
        self.num_classes = data.num_classes




        self.network = networks.clsdd(data.input_shape,hparams,data.num_classes,data.num_domain-1)

        self.network.cuda()
        state_dict = torch.load('./weights/model_CelebA_target_{}.ckpt'.format(fenv))
        self.network.load_state_dict(state_dict['model'], strict=True)



        self.network.eval()




        self.bs = 512
        self.k1 = k1
        self.k2 = k2
        self.envid = 0




        self.ordataset=data.ordataset






        self.datasets=[]

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
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





        for id,i in enumerate(self.ordataset):

            key1 = self.k1[id]
            key2 = self.k2[id]

            imgs = []
            ls = []
            ds = []
            ols=[]
            for filename, label, domain,orlabels in i:
                ls.append(label)
                ds.append(domain)
                ols.append(orlabels)
                image = self.read_image(os.path.join(data.image_dir, filename))
                image=transform(image)
                imgs.append(image)
            imgs=torch.stack(imgs)
            ls=torch.tensor(ls)
            ds=torch.tensor(ds)
            ols=torch.tensor(ols)

            xid = 0
            z = []
            zy=[]
            with torch.no_grad():
                while xid < imgs.shape[0]:
                    tmpx = imgs[xid:xid + self.bs]
                    tmpx = tmpx.cuda()
                    tmpp = self.network(tmpx)

                    z.extend(tmpp.tolist())

                    xid += self.bs

                z = torch.tensor(z)

                n = z.shape[0]


                keybool = [0 if i in key1 else 1 for i in range(n)]
                keybool = torch.tensor(keybool)

                # from IPython import embed
                # embed()

                # l or c
                z1 = z.norm(dim=1, keepdim=True).expand_as(z)

                z/=(z1+1e-6)






                distmat = torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                          torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n).t()
                distmat.addmm_(beta=1, alpha=-2, mat1=z, mat2=z.t())



                ye = ls.expand(n, n)
                ke = keybool.expand(n, n)
                k_notequal = (ke != ke.t())
                y_notequal = (ye != ye.t())
                distmat = distmat * y_notequal + distmat.max() * 2 * (~y_notequal) + distmat.max() * 2 * k_notequal
                index = distmat.min(0).indices

                # !!!!!



            ols=ols.long()
            ls=ls.long()
            ds=ds.long()





            weights = []
            for i_label in range(data.num_classes):
                i_label_num = (ls == i_label).float().mean()
                weights.append(i_label_num)

            # if id in test_envs:
            #     t=augment_transform
            # else:
            #     t=transform
            # self.datasets.append(TensorDataset_b_trans(imgs, ls, matchindex=index, weights=weights, keybool=keybool,
            #                                      inputshape=data.input_shape,transform=t))
            self.datasets.append(TensorDataset_b(imgs, ls, matchindex=index, weights=weights, keybool=keybool,
                                                 inputshape=data.input_shape))

        rs = []
        for i in self.datasets:
            rs.append((i.weights * i.n).sum())
        rsmin = np.array(rs).min()
        for id, i in enumerate(self.datasets):
            i.adjust_n(rsmin / rs[id])
            # from IPython import embed
            # embed()

    def read_image(self,img_path):
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
    CHECKPOINT_FREQ=300
    def __init__(self, root, test_envs, hparams,k1, k2,fenv,data):

        self.input_shape=data.input_shape
        self.num_classes = data.num_classes

        del data.datasets

        self.network = networks.clsdd(data.input_shape,hparams,data.num_classes,data.num_domain-1)

        self.network.cuda()
        state_dict = torch.load('./weights/model_CelebA_1_target_{}.ckpt'.format(fenv))
        self.network.load_state_dict(state_dict['model'], strict=True)



        self.network.eval()




        self.bs = 512
        self.k1 = k1
        self.k2 = k2
        self.envid = 0




        self.ordataset=data.ordataset






        self.datasets=[]

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
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





        for id,i in enumerate(self.ordataset):

            key1 = self.k1[id]
            key2 = self.k2[id]

            imgs = []
            ls = []
            ds = []
            ols=[]
            for filename, label, domain,orlabels in i:
                ls.append(label)
                ds.append(domain)
                ols.append(orlabels)
                image = self.read_image(os.path.join(data.image_dir, filename))
                image=transform(image)
                imgs.append(image)
            imgs=torch.stack(imgs)
            ls=torch.tensor(ls)
            ds=torch.tensor(ds)
            ols=torch.tensor(ols)

            xid = 0
            z = []
            zy=[]
            with torch.no_grad():
                while xid < imgs.shape[0]:
                    tmpx = imgs[xid:xid + self.bs]
                    tmpx = tmpx.cuda()
                    tmpp = self.network(tmpx)

                    z.extend(tmpp.tolist())

                    xid += self.bs

                z = torch.tensor(z)

                n = z.shape[0]


                keybool = [0 if i in key1 else 1 for i in range(n)]
                keybool = torch.tensor(keybool)

                # from IPython import embed
                # embed()

                # l or c
                z1 = z.norm(dim=1, keepdim=True).expand_as(z)

                #z/=(z1+1e-6)






                distmat = torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                          torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n).t()
                distmat.addmm_(beta=1, alpha=-2, mat1=z, mat2=z.t())



                ye = ls.expand(n, n)
                ke = keybool.expand(n, n)
                k_notequal = (ke != ke.t())
                y_notequal = (ye != ye.t())
                distmat = distmat * y_notequal + distmat.max() * 2 * (~y_notequal) + distmat.max() * 2 * k_notequal
                index = distmat.min(0).indices




            ols=ols.long()
            ls=ls.long()
            ds=ds.long()





            weights = []
            for i_label in range(data.num_classes):
                i_label_num = (ls == i_label).float().mean()
                weights.append(i_label_num)

            # if id in test_envs:
            #     t=augment_transform
            # else:
            #     t=transform
            # self.datasets.append(TensorDataset_b_trans(imgs, ls, matchindex=index, weights=weights, keybool=keybool,
            #                                      inputshape=data.input_shape,transform=t))
            self.datasets.append(TensorDataset_b(imgs, ls, matchindex=index, weights=weights, keybool=keybool,
                                                 inputshape=data.input_shape))

        rs = []
        for i in self.datasets:
            rs.append((i.weights * i.n).sum())
        rsmin = np.array(rs).min()
        for id, i in enumerate(self.datasets):
            i.adjust_n(rsmin / rs[id])
            # from IPython import embed
            # embed()

    def read_image(self,img_path):
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


class MultipleEnvironmentMNIST(MultipleDomainDataset):




    def __init__(self, root, environments, dataset_transform, input_shape,hparams,
                 num_classes,dataname,fenv,k1,k2):
        super().__init__()

        num_domain=len(environments)-1
        self.network = networks.clsdd(input_shape, hparams, num_classes, num_domain)
        # state_dict = torch.load('./modelbestce.ckpt')
        state_dict = torch.load('./weights/model_{}_target_{}.ckpt'.format(dataname, fenv))
        self.network.load_state_dict(state_dict['model'], strict=True)
        self.network.eval()
        self.network.cuda()
        self.bs = 128
        self.k1 = k1
        self.k2 = k2
        self.envid = 0


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

    def __init__(self, root, test_envs, hparams,k1, k2,fenv):

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), hparams,2,'ColoredMNIST',fenv,k1,k2)

        rs = []
        for i in self.datasets:
            rs.append((i.weights * i.n).sum())
        rsmin = np.array(rs).min()

        for id, i in enumerate(self.datasets):
            i.adjust_n(rsmin / rs[id])
            # from IPython import embed
            # embed()




    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        key1=self.k1[self.envid]
        key2=self.k2[self.envid]

        keybool=[0 if i in key1 else 1 for i in range(labels.shape[0])]
        keybool=torch.tensor(keybool)

        labels = (labels < 5).float()
        # Flip label with probability 0.25
        il = torch.ones_like(images) * (labels.unsqueeze(1).unsqueeze(2)) * 255
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel1
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        xid = 0
        z = []
        with torch.no_grad():
            while xid < x.shape[0]:
                tmpx = x[xid:xid + self.bs]
                tmpx = tmpx.cuda()
                tmpp = self.network(tmpx)
                z.extend(tmpp.tolist())
                xid += self.bs

            z=torch.tensor(z)
            n = z.shape[0]

            z1 = z.norm(dim=1, keepdim=True).expand_as(z)
            z=z/(1e-6+z1)


            distmat = torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                      torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n).t()
            distmat.addmm_(beta=1, alpha=-2, mat1=z, mat2=z.t())
            ye = y.expand(n, n)
            ke=keybool.expand(n, n)
            k_notequal=( ke != ke.t())
            y_notequal =( ye != ye.t())
            distmat=distmat*y_notequal+distmat.max()*2*(~y_notequal)+distmat.max()*2*k_notequal
            index=distmat.min(0).indices



            weights = []
            for i_label in range(self.num_classes):
                i_label_num = (y == i_label).float().mean()
                weights.append(i_label_num)

        # if environment==0.9:
        #     return TensorDataset(x, y)

        self.envid+=1
        #return TensorDataset(x, y, x[index],y[index],index)

        return TensorDataset_b(x, y,matchindex=index,weights=weights,keybool=keybool,inputshape=self.input_shape)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()




class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams,k1, k2,fenv):
        self.input_shape = (1, 28, 28,)
        self.num_classes = 10


        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,),hparams, 10,'RotatedMNIST',fenv, k1, k2)

        rs = []
        for i in self.datasets:
            rs.append((i.weights * i.n).sum())
        rsmin = np.array(rs).min()

        for id, i in enumerate(self.datasets):
            i.adjust_n(rsmin / rs[id])


    def rotate_dataset(self, images, labels, angle):
        key1 = self.k1[self.envid]
        key2 = self.k2[self.envid]

        keybool = [0 if i in key1 else 1 for i in range(labels.shape[0])]
        keybool = torch.tensor(keybool)
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        xid = 0
        z = []
        with torch.no_grad():
            while xid < x.shape[0]:
                tmpx = x[xid:xid + self.bs]
                tmpx = tmpx.cuda()
                tmpp = self.network(tmpx)
                z.extend(tmpp.tolist())
                xid += self.bs

            z = torch.tensor(z)
            n = z.shape[0]

            z1 = z.norm(dim=1, keepdim=True).expand_as(z)
            z = z / (1e-6 + z1)

            distmat = torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                      torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n).t()
            distmat.addmm_(beta=1, alpha=-2, mat1=z, mat2=z.t())
            ye = y.expand(n, n)
            ke = keybool.expand(n, n)
            k_notequal = (ke != ke.t())
            y_notequal = (ye != ye.t())
            distmat = distmat * y_notequal + distmat.max() * 2 * (~y_notequal) + distmat.max() * 2 * k_notequal
            index = distmat.min(0).indices

            weights = []
            for i_label in range(self.num_classes):
                i_label_num = (y == i_label).float().mean()
                weights.append(i_label_num)

            # if environment==0.9:
            #     return TensorDataset(x, y)

        self.envid += 1
        # return TensorDataset(x, y, x[index],y[index],index)

        return TensorDataset_b(x, y, matchindex=index, weights=weights, keybool=keybool, inputshape=self.input_shape)


class TensorDataset_b_trans(Dataset):
    def __init__(self, *tensors,matchindex,weights,keybool,inputshape,transform=None):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.transform=transform
        self.tensors = tensors
        self.matchindex = matchindex
        weights = np.array(weights)
        self.weights = weights
        self.keybool = keybool
        n = (1 - weights) / weights
        n = n / n.max()

        self.n = n
        self.null_ind = torch.tensor(0)
        self.null_sample = torch.zeros(inputshape)
        self.null_target = torch.tensor(-1)
        self.sr = np.random.rand(len(matchindex))

    def __getitemnb__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def adjust_n(self,ratio):
        self.n=self.n*ratio

    def read_image(self,img_path):
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


        # if self.transform is not None:
        #     sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        if self.keybool[index]:
            rr=random.random()
        else:
            rr=self.sr[index]
        if rr<self.n[target]:
            mi = self.matchindex[index]
            pathi, targeti = self.__getitemnb__(mi)
            samplei=self.read_image(pathi)
            if self.transform is not None:
                samplei = self.transform(samplei)

        else:

            return sample,target,self.null_sample,self.null_target,self.null_ind


        return sample, target,samplei, targeti,mi

    def __len__(self):
        return self.tensors[0].size(0)

class TensorDataset_b(Dataset):
    def __init__(self, *tensors,matchindex,weights,keybool,inputshape):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.matchindex = matchindex
        weights = np.array(weights)
        self.weights = weights
        self.keybool = keybool
        n = (1 - weights) / weights
        n = n / n.max()

        self.n = n
        self.null_ind = torch.tensor(0)
        self.null_sample = torch.zeros(inputshape)
        self.null_target = torch.tensor(-1)
        self.sr = np.random.rand(len(matchindex))

    def __getitemnb__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def adjust_n(self,ratio):
        self.n=self.n*ratio

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        sample,target = self.__getitemnb__(index)


        # if self.transform is not None:
        #     sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        if self.keybool[index]:
            rr=random.random()
        else:
            rr=self.sr[index]
        if rr<self.n[target]:
            mi = self.matchindex[index]
            samplei, targeti = self.__getitemnb__(mi)

        else:

            return sample,target,self.null_sample,self.null_target,self.null_ind


        return sample, target,samplei, targeti,mi

    def __len__(self):
        return self.tensors[0].size(0)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams,k1,k2,dataname,inputshape,num_classes,num_domain,fenv):
        super().__init__()

        # from IPython import embed
        # embed()

        self.network = networks.clsdd(inputshape,hparams,num_classes,num_domain)
        #state_dict = torch.load('./modelbestce.ckpt')
        state_dict = torch.load('./weights/model_{}_target_{}.ckpt'.format(dataname,fenv))
        self.network.load_state_dict(state_dict['model'], strict=True)
        self.network.eval()
        self.network.cuda()
        self.bs = 128
        self.k1 = k1
        self.k2 = k2
        self.envid = 0

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

            key1 = self.k1[i]
            key2 = self.k2[i]

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            # env_dataset = ImageFolder(path,
            #     transform=env_transform)

            tmp=ImageFolder(path, transform=transform)
            imgs=[]
            ids=[]
            z = []
            for jd, j in enumerate(tmp):
                img,id=j[0],j[1]
                imgs.append(img)
                ids.append(id)
                if len(imgs)>=self.bs:
                    with torch.no_grad():
                        tmpx=torch.stack(imgs)
                        tmpx = tmpx.cuda()
                        tmpp = self.network(tmpx)
                        z.extend(tmpp.tolist())
                        imgs=[]

            with torch.no_grad():
                if len(imgs)>0:
                    tmpx = torch.stack(imgs)
                    tmpx = tmpx.cuda()
                    tmpp = self.network(tmpx)
                    z.extend(tmpp.tolist())
                    imgs = []

                #imgs = torch.stack(imgs)
                ls = torch.tensor(ids)




            # xid = 0
            # z = []
            #with torch.no_grad():
                # while xid < imgs.shape[0]:
                #     tmpx = imgs[xid:xid + self.bs]
                #     tmpx = tmpx.cuda()
                #     tmpp = self.network(tmpx)
                #     z.extend(tmpp.tolist())
                #     xid += self.bs

                z = torch.tensor(z)
                n = z.shape[0]

                keybool = [0 if i in key1 else 1 for i in range(n)]
                keybool = torch.tensor(keybool)

                z1 = z.norm(dim=1, keepdim=True).expand_as(z)
                z=z/(1e-6+z1)

                # from IPython import embed
                # embed()

                distmat = torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                          torch.pow(z, 2).sum(dim=1, keepdim=True).expand(n, n).t()
                distmat.addmm_(beta=1, alpha=-2, mat1=z, mat2=z.t())
                ye = ls.expand(n, n)
                ke = keybool.expand(n, n)
                k_notequal = (ke != ke.t())
                y_notequal = (ye != ye.t())
                distmat = distmat * y_notequal + distmat.max() * 2 * (~y_notequal) + distmat.max() * 2 * k_notequal
                index = distmat.min(0).indices

                weights=[]
                for i_label in range(num_classes):
                    i_label_num = (ls == i_label).float().mean()
                    weights.append(i_label_num)
                #print('weights',weights)
                #
                # mweights = []
                # for i_label in range(num_classes):
                #     i_label_num = (ls[index] == i_label).float().mean()
                #     mweights.append(i_label_num)
                # print('mweights', mweights)
                #
                # from IPython import embed
                # embed()



            env_dataset = newImageFolder0(path,
                transform=env_transform,matchindex=index,weights=weights,keybool=keybool,inputshape=inputshape)

            self.datasets.append(env_dataset)
        rs=[]
        for i in self.datasets:
            rs.append((i.weights*i.n).sum())
        rsmin=np.array(rs).min()

        for id,i in enumerate(self.datasets):
            i.adjust_n(rsmin/rs[id])
            # from IPython import embed
            # embed()


        del self.network
        del tmpp
        del tmpx

        torch.cuda.empty_cache()


        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class newImageFolder0(ImageFolder):
    def __init__(self,path,transform,matchindex,weights,keybool,inputshape):
        super(newImageFolder0, self).__init__(path,transform)
        self.matchindex=matchindex
        weights=np.array(weights)
        self.weights=weights
        self.keybool=keybool
        n = (1 - weights) / weights
        n = n / n.max()
        self.n=n
        self.null_ind=torch.tensor(0)
        self.null_sample=torch.zeros(inputshape)
        self.null_target=-1
        self.sr=np.random.rand(len(matchindex))

    def adjust_n(self,ratio):
        self.n=self.n*ratio

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.keybool[index]:
            rr=random.random()
        else:
            rr=self.sr[index]
        if rr<self.n[target]:
            mi = self.matchindex[index]
            pathi, targeti = self.samples[mi]
            samplei = self.loader(pathi)
            if self.transform is not None:
                samplei = self.transform(samplei)
            if self.target_transform is not None:
                targeti = self.target_transform(targeti)
        else:
            return sample,target,self.null_sample,self.null_target,self.null_ind

        return sample, target,samplei, targeti,mi

    def __len__(self) -> int:
        return len(self.samples)

class newImageFolder(ImageFolder):
    def __init__(self,path,transform,matchindex):
        super(newImageFolder, self).__init__(path,transform)
        self.matchindex=matchindex
        self.loadimgs=[]
        for index,i in enumerate(self.samples):
            path=i[0]
            sample = self.loader(path)
            self.loadimgs.append(sample)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        mi=self.matchindex[index]
        pathi, targeti = self.samples[mi]
        # sample = self.loader(path)
        # samplei = self.loader(pathi)
        sample = self.loadimgs[index]
        samplei = self.loadimgs[mi]

        if self.transform is not None:
            sample = self.transform(sample)
            samplei = self.transform(samplei)
        if self.target_transform is not None:
            target = self.target_transform(target)
            targeti = self.target_transform(targeti)

        return sample, target,samplei, targeti,mi

    def __len__(self) -> int:
        return len(self.samples)








class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams,k1, k2,fenv):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams,k1, k2, dataname='VLCS', inputshape=(3,224,224), num_classes=5, num_domain=3,fenv=fenv)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams,k1, k2,fenv):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams,k1, k2, dataname='PACS', inputshape=(3,224,224), num_classes=7, num_domain=3,fenv=fenv)



class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams,k1, k2,fenv):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams,k1, k2, dataname='DomainNet', inputshape=(3,224,224), num_classes=345, num_domain=5,fenv=fenv)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams,k1, k2,fenv):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams,k1, k2, dataname='OfficeHome', inputshape=(3,224,224), num_classes=65, num_domain=3,fenv=fenv)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams,k1, k2,fenv):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams,k1, k2, dataname='TerraIncognita', inputshape=(3,224,224), num_classes=10, num_domain=3,fenv=fenv)

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

