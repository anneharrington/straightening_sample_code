import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import timm
import torchvision
import torchvision.models as models
from torchvision import transforms
import os
import sys
from timm.models import create_model
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
from fast_curvature import *

# parameters
path = 1
model_arch = 'crossvit'
training = 'robust'
output_dir = f'./{model_arch}_curves'
name = f'{model_arch}_{training}_august_path_{str(path)}'

if training == 'baseline':
    pretrained = True
elif training == 'robust':
    pretrained = False
if model_arch == 'crossvit':
    xlabels = ['Pixel', 'Block1', 'Block2', 'Block3', 'Prediction']
    model = create_model('crossvit_18_dagger_408',pretrained = pretrained)
    if not pretrained:
        print("loading checkpoint")
        checkpoint = torch.load('../../checkpoint_epoch5.pt')
        model.load_state_dict(checkpoint['state_dict'])
        print("finished loading")
elif model_arch == 'cnn':
    path = None
    xlabels = ['Pixel', 'Layer1', 'Layer2', 'Layer3', 'Layer4', 'Avgpool', 'Prediction']
    model = models.__dict__['resnet50'](pretrained=pretrained)
    if not pretrained:
        print("loading checkpoint")
        checkpoint = torch.load('imagenet_l2_3_0.pt')
        from collections import OrderedDict
        new_state_dict1 = OrderedDict()
        i= 0
        for k, v in checkpoint['model'].items():
            if i == 0 or i == 1:
                i = i + 1
                continue
            if i>321:
                i = i + 1
                continue
            name_r =  k[13:] # remove `module.`
            new_state_dict1[name_r] = v
            i = i + 1

        model.load_state_dict(new_state_dict1)
        print("finished loading")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = nn.DataParallel(model) # crossvit need extra memory
model = model.to(device)
model.eval()

if model_arch == 'crossvit':
    model_blocks = [model.module.blocks[0],model.module.blocks[1],model.module.blocks[2]]
elif model_arch == 'cnn':
    model_blocks = [model.module.layer1,
                model.module.layer2,
                model.module.layer3,
                model.module.layer4,
                model.module.avgpool]

# videos
vid = load_all_henaff_videos('./stimuli/gamma1/',img_size=256)
natural = torch.from_numpy(vid['natural'],).permute((0,1,4,3,2)).float().contiguous()
contrast = torch.from_numpy(vid['contrast']).permute((0,1,4,3,2)).float().contiguous()
artificial = torch.from_numpy(vid['artificial']).permute((0,1,4,3,2)).float().contiguous()
videos = [natural.to(device),contrast.to(device), artificial.to(device)]
all_videos = vid

# curvature
curvs_natural = get_intermediate_curv(model, model_blocks,videos[0],vit=True,path=path,)
curvs_contrast = get_intermediate_curv(model, model_blocks,videos[1],vit=True,path=path)
curvs_artificial = get_intermediate_curv(model, model_blocks,videos[2],vit=True,path=path)

mean_natural = curvs_natural.mean(0)
mean_contrast = curvs_contrast.mean(0)
mean_artificial = curvs_artificial.mean(0)

# standard error
std_natural = curvs_natural.std(0)/curvs_natural.size(0)**0.5
std_contrast = curvs_contrast.std(0)/curvs_contrast.size(0)**0.5
std_artificial = curvs_artificial.std(0)/curvs_artificial.size(0)**0.5

natural_plot = mean_natural - mean_natural[0].repeat(mean_natural.size(0))
print(natural_plot)
contrast_plot = mean_contrast - mean_contrast[0].repeat(mean_contrast.size(0))
artificial_plot = mean_artificial - mean_artificial[0].repeat(mean_artificial.size(0))

# plot
os.makedirs(output_dir,exist_ok=True)

plt.figure(figsize=(6,4))
plt.errorbar(xlabels, natural_plot, yerr=std_natural, marker = 'o', linestyle='--',markersize=8,label='natural',uplims=True, lolims=True)
plt.errorbar(xlabels, contrast_plot, yerr=std_contrast, marker = 'o', linestyle='--',markersize=8,label='contrast',uplims=True, lolims=True)
plt.errorbar(xlabels, artificial_plot, yerr=std_artificial, marker = 'o', linestyle='--',markersize=8,label='artificial',uplims=True, lolims=True)

plt.plot([xlabels[0], xlabels[-1]],[0,0], '-', color='gray')
plt.ylim([-90, 90])
plt.ylabel('Change in Mean Curvature($^\circ$)')
plt.text(xlabels[-2], -80,'lower curvature', color='gray',fontsize='small')
plt.text(xlabels[-2], 80,'higher curvature', color='gray',fontsize='small')
plt.legend(loc='lower left')
plt.tight_layout()
save_name = os.path.join(output_dir,f'{name}_curve_change.svg')
plt.savefig(save_name)

save_name = os.path.join(output_dir,f'{name}_curve_change.png')
plt.savefig(save_name,dpi=300)
print('done!')