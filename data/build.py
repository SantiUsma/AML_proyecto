# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset,TensorDataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .cached_image_folder import CachedImageFolder
from .imagenet22k_dataset import IN22KDATASET
from .samplers import SubsetRandomSampler
from glob import glob
from skimage import io
import cv2
from tqdm import tqdm
import pickle as pkl

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_loader(config,mode='train',imagen_demo=False):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config,mode=mode,imagen_demo=imagen_demo)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config,mode=mode,imagen_demo=imagen_demo)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    if mode=='demo':
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=1,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
        )
    if mode=='demo':
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )
    else:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )
    

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config,mode='train',imagen_demo=False):
    def PositionalEncoding(size, value):
            d=1
            for i in range(len(size)):
                d=d*size[i]
            j=int(d/2)
            PE=np.zeros(d)
            for i in range(j):
                PE[2*i] = np.sin(value/(10000**(2*i/d)))
                PE[2*i + 1] = np.cos(value / (10000**(2*i/d)))
            if d%2 !=0:

                PE[2*j] = np.sin(value / (10000 ** (2 * j / d)))

            PE=np.reshape(PE,size)
            return PE
    pos1=PositionalEncoding((1024,1024,1),int(1))
    pos2=PositionalEncoding((1024,1024,1),int(2))
    pos3=PositionalEncoding((1024,1024,1),int(3))
    transform = build_transform(is_train, config)
    
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            if mode=='demo':
                assert imagen_demo!=False, 'No se proporciono una imagen de prueba'
                imagen=np.zeros((1,4,1024,1024))
                target=np.zeros(1)
                edades=np.zeros(1)
                im=io.imread(imagen_demo)
                im=cv2.resize(im,(1024,1024))
                if len(im.shape)!=3:
                        im=np.stack((im,im,im),axis=2)
                ed=int(imagen_demo.split('/')[-1].split('.')[1])
                if int(ed)==1:
                    im=np.concatenate((im,pos1),axis=2)
                if int(ed)==2:
                    im=np.concatenate((im,pos2),axis=2)
                if int(ed)==3:
                    im=np.concatenate((im,pos3),axis=2)
                im=np.transpose(im,axes=(2,0,1))
                imagen[0]=im
                images_all=torch.Tensor(imagen)
                targets_all=torch.Tensor(target).type(torch.LongTensor)
                edad_all=torch.as_tensor(edades)
                dataset=TensorDataset(images_all,targets_all,edad_all)
            else:
                root_N = os.path.join(config.DATA.DATA_PATH, prefix,'Negative','*.jpeg')
                root_P = os.path.join(config.DATA.DATA_PATH, prefix,'Positive','*.jpeg')
                paths_N=glob(root_N)
                paths_P=glob(root_P)
                imagenes_P=np.zeros((len(paths_P),4,1024,1024))
                imagenes_N=np.zeros((len(paths_N),4,1024,1024))
                targets_P=np.zeros(len(paths_P))
                targets_N=np.zeros(len(paths_N))
                edades_N=np.zeros(len(paths_N))
                edades_P=np.zeros(len(paths_P))
                names_n=[]
                names_p=[]
                for i in tqdm(range(len(paths_N))):
                    im=io.imread(paths_N[i])
                    im=cv2.resize(im,(1024,1024))
                    if len(im.shape)!=3:
                        im=np.stack((im,im,im),axis=2)
                    ed=int(paths_N[i].split('/')[-1].split('.')[1])
                    if int(ed)==1:
                        im=np.concatenate((im,pos1),axis=2)
                    if int(ed)==2:
                        im=np.concatenate((im,pos2),axis=2)
                    if int(ed)==3:
                        im=np.concatenate((im,pos3),axis=2)
                    im=np.transpose(im,axes=(2,0,1))
                    imagenes_N[i,:,:,:]=im
                    targets_N[i]=int(0)
                    edades_N[i]=int(ed)
                    names_n.append(paths_N[i])

                for i in tqdm(range(len(paths_P))):
                    im=io.imread(paths_P[i])
                    im=cv2.resize(im,(1024,1024))
                    ed=paths_P[i].split('/')[-1].split('.')[1]
                    if len(im.shape)!=3:
                        im=np.stack((im,im,im),axis=2)
                    if int(ed)==1:
                        im=np.concatenate((im,pos1),axis=2)
                    if int(ed)==2:
                        im=np.concatenate((im,pos2),axis=2)
                    if int(ed)==3:
                        im=np.concatenate((im,pos3),axis=2)
                    im=np.transpose(im,axes=(2,0,1))
                    imagenes_P[i,:,:,:]=im
                    targets_P[i]=int(1)
                    edades_P[i]=int(ed)
                    names_p.append(paths_P[i])
                paths_all=np.concatenate((names_n,names_p),axis=0)
                images_all=np.concatenate((imagenes_N,imagenes_P),axis=0)
                targets_all=np.concatenate((targets_N,targets_P),axis=0)
                images_all=torch.Tensor(images_all)
                targets_all=torch.Tensor(targets_all).type(torch.LongTensor)
                edad_all=torch.as_tensor(np.concatenate((edades_N,edades_P),axis=0))
                #dataset = datasets.ImageFolder(root, transform=transform)
                with open('paths.mat','wb') as f:
                    pkl.dump(paths_all,f)
                dataset=TensorDataset(images_all,targets_all,edad_all)
        nb_classes = 2
    elif config.DATA.DATASET == 'imagenet22K':
        prefix = 'ILSVRC2011fall_whole'
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
        nb_classes = 2
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    #t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
