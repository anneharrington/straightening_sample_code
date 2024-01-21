# plot curvature of walking video and plot trajectory
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import os
import math
from PIL import Image
from utils import computeDistCurv

# load walking videos
im_dir = './stimuli/gamma1/walking'
PILtoTensor = torchvision.transforms.ToTensor()
walking_frames = torch.zeros((11,1,512,512))
for i in range(11):
    walking_frames[i,:,:,:] = PILtoTensor(Image.open(os.path.join(im_dir, 'groundtruth' + str(i+1) + '.png')))

# plot walking view sequence
fig, axes = plt.subplots(1,11, figsize=(20,12))
for i in range(11):
    axes[i].imshow(torch.squeeze(walking_frames[i,:,:]), cmap='gray', vmin=0, vmax=1)
    axes[i].grid(False)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.savefig('walking_frames.png')

# performing PCA and projecting to first two componenets
walking_flattened = walking_frames.view( walking_frames.size(0), -1 ) 
# mean for centering
walking_mean = torch.mean(walking_flattened,axis=0)
walking_centered = walking_flattened - walking_mean
# PCA
U,S,V = torch.pca_lowrank(walking_centered, center=False, q=walking_frames.size(0))
walking_projected = torch.matmul(walking_centered, V[:, 0:2])

# plot trajectory in PC space
plt.figure()
plt.plot(walking_projected[:,0], walking_projected[:,1], 'o-')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Pixel Intensity Space Trajectory')
plt.savefig('walking_pca_trajectory.png')

# plot Pixel vs PCA curvature
plt.figure()
plt.plot(computeDistCurv(walking_projected)[1],'.-', label='pca')
plt.plot(computeDistCurv(walking_frames)[1],'.-',label='pixel')
plt.ylim([-1, 180])
plt.title('Curvature between frames')
plt.ylabel('degrees')
plt.xlabel('between frames (time)')
plt.legend()
plt.savefig('walking_curvature.png')

# get global curvature
print('projected', 'max=' + str(torch.max(computeDistCurv(walking_projected)[1]).item()), 'mean=' + str(torch.mean(computeDistCurv(walking_projected)[1]).item()))
print('raw pixel', 'max=' + str(torch.max(computeDistCurv(walking_frames)[1]).item()), 'mean= ' + str(torch.mean(computeDistCurv(walking_frames)[1]).item()))

