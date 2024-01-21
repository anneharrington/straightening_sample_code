import os
import numpy as np 
# import utilities as utils 
# import params 
import torch
from utils_henaff import *
import sys

print('starting')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# setting path
sys.path.append('../../straightening_models_code/Perceptual-Straightening-models/')
import utilities as utils
from straightening_hierarchy import Straightening_Hierarchy
from models.steerable.config import device
from torch.autograd import Variable

vid = load_all_henaff_videos_corrected(rgb=False,img_size=256)
natural = vid['natural'].float().contiguous()
artificial = vid['artificial'].float().contiguous()
contrast = vid['contrast'].float().contiguous()
videos = [natural,contrast, artificial]
print(natural.max())


all_videos = vid

modelStages = ['pixel', 'retina', 'v1']


nparams = 6
kparams = 8
print('creating model!')
model = Straightening_Hierarchy(256, N=nparams, K=kparams)
model.to(device)
print('model created!')

curve_outputs = []
for s in range(3):
    current_videos = videos[s].squeeze()
    video_curves = torch.zeros(len(current_videos),3)

    for v,vid in enumerate(current_videos):
        vid.to(device)
        vid = Variable(vid.clone()) 
        vid = vid.to(device)
        y = model( vid*255. )
        for m, modelStage in enumerate(modelStages):
            dY, cY = computeDistCurv( y[modelStage] )
            # print(s,v,m,cY)
            video_curves[v,m] = cY.data.mean().item()
            # todo: save the std

    curve_outputs.append(video_curves.mean(0))

print([torch.round(o) for o in curve_outputs])

mean_natural = curve_outputs[0]
mean_contrast = curve_outputs[1]
mean_artificial = curve_outputs[2]


natural_plot = mean_natural - mean_natural[0].repeat(mean_natural.size(0))
contrast_plot = mean_contrast - mean_contrast[0].repeat(mean_contrast.size(0))
artificial_plot = mean_artificial - mean_artificial[0].repeat(mean_artificial.size(0))



xlabels = ['Pixel', 'Retina/LGN', 'V1']
output_dir = './henaffbio_curves'
os.makedirs(output_dir,exist_ok=True)

all_curves = {'layer_names':xlabels,'natural_curves':mean_natural, 'contrast_curves':mean_contrast,'artificial_curves':mean_artificial,
              'natural_ste':torch.ones_like(curve_outputs[0])*float('NaN'), 'contrast_ste':torch.ones_like(curve_outputs[0])*float('NaN'),'artificial_ste':torch.ones_like(curve_outputs[0])*float('NaN')}

plt.figure(figsize=(10,4))
plt.plot(xlabels, natural_plot, marker = 'o', linestyle='--',markersize=8,label='natural')
plt.plot(xlabels, contrast_plot, marker = 'o', linestyle='--',markersize=8,label='contrast')
plt.plot(xlabels, artificial_plot, marker = 'o', linestyle='--',markersize=8,label='artificial')

plt.plot([xlabels[0], xlabels[-1]],[0,0], '-', color='gray')
plt.ylim([-60, 60])
plt.ylabel('Change in Mean Curvature($^\circ$)')
plt.text(xlabels[2], -80,'lower curvature', color='gray',fontsize='small')
plt.text(xlabels[2], 80,'higher curvature', color='gray',fontsize='small')
plt.legend(loc='lower left')
# plt.title('Change in Curvature Across Alexnet Blocks (with PCA)', fontsize=12)
plt.tight_layout()
name = 'henaffbio'
save_name = os.path.join(output_dir,f'{name}_curve_change.png')
plt.savefig(save_name,dpi=300)
write_model_csv(output_dir, name, all_curves)

