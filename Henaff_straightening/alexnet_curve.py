# plot change in curvature across Alexnet blocks
# for one video

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from utils import computeDistCurv

def torch_pca_projection(x, num_pcs=2):
    x_flat = x.view(x.size(0), -1 )
    x_mean = torch.mean(x_flat,axis=0)
    x_centered = x_flat - x_mean
    U,S,V = torch.pca_lowrank(x_centered, center=False, q=x.size(0))
    x_projected = torch.matmul(x_centered, V[:, 0:num_pcs])
    return x_projected, x_mean

def get_model_curve(blocks, video, pca=False):  
    block_output = [block(video) for block in blocks]
    if pca:
        model_output = []
        for block in block_output:
            block_projected = torch_pca_projection(block,2)[0]
            model_output.append(block_projected)
        input_vid = torch_pca_projection(video)[0]
    else:
        model_output = block_output
        input_vid = video
    return [computeDistCurv(block_out)[1] for block_out in model_output], computeDistCurv(input_vid)[1]

def get_curve_change(model_curve, video_curve):
    curve_change = [0]*(len(model_curve) + 1)
    for c in range(len(model_curve)):
        curve_change[c+1] = (torch.mean(model_curve[c] - video_curve)).item()
    print(curve_change)
    return curve_change

# load model
name = 'alexnet'
model = models.__dict__[name](pretrained=True)

model.eval()

# get blocks
pool1 = torch.nn.Sequential(*list(model.features.children())[:3])
pool2 = torch.nn.Sequential(*list(model.features.children())[:6])
pool3 = torch.nn.Sequential(*list(model.features.children())[:])

pool1.eval()
pool2.eval()
pool3.eval()

model_blocks = [pool1, pool2, pool3]

# get walking video
im_dir = './stimuli/gamma1/walking'
PILtoTensor = torchvision.transforms.ToTensor()
walking_frames = torch.zeros((11,3,512,512))
for i in range(11):
    img = Image.open(os.path.join(im_dir, 'groundtruth' + str(i+1) + '.png'))
    rgbimg = img.convert('RGB')
    walking_frames[i,:,:,:] = PILtoTensor(rgbimg)

model_curve_orig, video_curve_orig = get_model_curve(model_blocks, walking_frames, pca=False)
model_curve_pca, video_curve_pca = get_model_curve(model_blocks, walking_frames, pca=True)

curve_change_orig = get_curve_change(model_curve_orig, video_curve_orig)
curve_change_pca = get_curve_change(model_curve_pca, video_curve_pca)
    
# plot change in curvature no pca
xlabels = ['Pixel', 'Block 1', 'Block 2', 'Block 3']
plt.figure()
plt.plot(xlabels, curve_change_orig, 'o')
plt.plot([xlabels[0], xlabels[-1]],[0,0], '--', color='gray')
plt.ylim([-20, 40])
plt.ylabel('Degrees')
plt.title('Change in Curvature Across Alexnet Blocks', fontsize=14)
plt.tight_layout()
plt.savefig(f'{name}_curve.png',dpi=300)

# plot change in curvature with pca
# the paper figure is an average over 12 videos
# the curvature actually flucctuates a lot across videos
# add error bars when averaging over many vidoes

xlabels = ['Pixel', 'Block 1', 'Block 2', 'Block 3']
plt.figure()
plt.plot(xlabels, curve_change_orig, 'o', label='no pca')
plt.plot(xlabels, curve_change_pca, 'o', label='pca')
plt.legend()
plt.plot([xlabels[0], xlabels[-1]],[0,0], '--', color='gray')
plt.ylim([-20, 40])
plt.ylabel('Degrees')
plt.title('Change in Curvature Across Alexnet Blocks', fontsize=14)
plt.tight_layout()
plt.savefig(f'{name}_curve_pca.png',dpi=300)
