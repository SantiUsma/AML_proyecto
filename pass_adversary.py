from glob import glob
import torch
from PIL import Image
import numpy as np

path="/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/adversarial_examples/fold2/Negative/*.pt"
path2="/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_adv_fold1_original/val/Negative/"

real_names_N=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_fold1_original/val/Negative/"+'*.jpeg')
real_names_P=glob("/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_fold1_original/val/Positive/"+'*.jpeg')

paths=glob(path)

for i in paths:
    l=i.split('/')[-1].split('.')
    l[0]=l[0]+'.'
    l[1]=l[1]+'.'
    l[2]=''
    l[3]=l[3]+'_adv'
    l[4]=''
    l=''.join(l)
    dato=torch.load(i,map_location='cpu')
    im=Image.fromarray(np.array(dato.permute(1, 2, 0).cpu().numpy(),dtype=np.uint8))
    print(path2+l+'.jpeg')
    im.save(path2+l+'.jpeg')

path="/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/adversarial_examples/fold1/Positive/*.pt"
path2="/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_adv_fold2_original/val/Positive/"
paths=glob(path)

for i in paths:
    l=i.split('/')[-1].split('.')
    l[0]=l[0]+'.'
    l[1]=l[1]+'.'
    l[2]=''
    l[3]=l[3]+'_adv'
    l[4]=''
    l=''.join(l)
    dato=torch.load(i,map_location='cpu')
    im=Image.fromarray(np.array(dato.permute(1, 2, 0).cpu().numpy(),dtype=np.uint8))
    print(path2+l+'.jpeg')
    im.save(path2+l+'.jpeg')