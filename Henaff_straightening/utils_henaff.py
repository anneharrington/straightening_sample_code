import torch
import torchvision
from PIL import Image
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import utilities as utilities
# sys.path.append('./../../straightening_models_code/Perceptual-Straightening-models')


# get curvature
def computeDistCurv( x ):
    # flatten input
    x = x.view( x.size(0), -1 )
    
    # displacemnt between time steps
    v = x[1:] - x[0:-1]
    d = (v**2).sum(1).sqrt()
    v.div_( d.unsqueeze(1) )
    
    # angle between displacement vectors
    c = ( v[1:] * v[0:-1] ).sum(1).clamp( -1, 1 ).acos()
    c.mul_( 180 / math.pi )

    return d, c 

def computeDistCurv_batch( x, grad=False ):
    """
    Given a batch of video frames (Tensor),
    Returns a the curvature across each video in the batch.
    
    'd' in the original function is replaced by torch.nn.funtional.normalize
    for gradient calculation
    """
    # flatten input
    x = x.view( x.size(0),x.size(1), -1 )
    
    # displacemnt between time steps
    v = x[:,1:] - x[:,0:-1]
    if grad:
        v = torch.nn.functional.normalize(v,dim=2)
    else:
        d = (v**2).sum(2).sqrt()
        v.div_( d.unsqueeze(2) )
    
    # angle between displacement vectors
    c = ( v[:,1:] * v[:,0:-1] ).sum(2).clamp( -1, 1 ).acos()
    c.mul_( 180 / math.pi )
    return c

# do PCA
def torch_pca_projection(x, num_pcs=2):
    x_flat = x.view(x.size(0), -1 )
    x_mean = torch.mean(x_flat,axis=0)
    x_centered = x_flat - x_mean
    U,S,V = torch.pca_lowrank(x_centered, center=False, q=x.size(0))
    x_projected = torch.matmul(x_centered, V[:, 0:num_pcs])
    return x_projected, V, x_mean

# get curvature for different 
def get_model_curve(blocks, video, pca=False, use_outputs=False,vit=False):
    """
    Given:
        blocks (list of model sections (each a model)), 
        video (tensor of shape (frames, channels, height, width)),
        pca (flag indicating if pca should be done before curvature analysis)
        use_outputs (flag for inputing 'blocks' as model outputs rather than model sections)
    
    Returns:
        (a list of curvature values for each model section,
        the curvature of the input pixel video)
    """
    if use_outputs:
        block_output = blocks
    else:
        if vit:
            block_output = blocks[0](video)
            block_output = [block_output[0], block_output[1]]
        else:
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

# difference in curvature
def get_curve_change(model_curve, video_curve):
    curve_change = [0]*(len(model_curve) + 1)
    for c in range(len(model_curve)):
        curve_change[c+1] = (torch.mean(model_curve[c] - video_curve)).item()
    return curve_change

#load videos
def load_henaff_video(vid_dir, natural=True, rgb=False,img_size=512):
    PILtoTensor = torchvision.transforms.ToTensor()
    if rgb:
        vid_frames = torch.empty((11,3,img_size,img_size))
    else:
        vid_frames = torch.empty((11,1,img_size,img_size))
    for i in range(11):
        if natural:
            frame = Image.open(os.path.join(vid_dir, 'groundtruth' + str(i+1) + '.png'))
        else:
            frame = Image.open(os.path.join(vid_dir, 'pixelfade' + str(i+1) + '.png'))
        if rgb:
            frame = frame.convert('RGB')
        frame = frame.resize((img_size,img_size))
        vid_frames[i,:,:,:] = PILtoTensor(frame)
    return vid_frames

def load_all_henaff_videos(videos_path = "/home/gridsan/groups/RosenholtzLab/straightening_models_code/Perceptual-Straightening-models/stimuli",rgb=True,img_size=512):
    """
    Returns a dictionary.  Each key is a video sequence type.  Each entry is a numpy array containing all videos of that type
    """
    video_types = ['natural', 'contrast','artificial']
    all_videos = {}
    for vid_type in video_types:
        print(vid_type)
        if vid_type == 'contrast':
            video_names = [ 'water-contrast0.5', 'prairieTer-contrastLog0.1', 'boats_contrastCocktail', 'bees_contrastCocktail', 'walking_contrastCocktail', 'egomotion_contrastCocktail', 'smile-contrastLog0.1', 'walking-contrast0.5', 'bees-contrast0.5', 'walking-contrastLog0.1' ]
        else:
            video_names = [ 'water', 'prairieTer', 'boats', 'ice3', 'dogville', 'egomotion', 'walking', 'smile', 'bees', 'leaves-wind', 'carnegie-dam', 'chironomus' ]

        if rgb:
            video_group = torch.empty((len(video_names), 11, img_size,img_size, 3))
        else:
            video_group = torch.empty((len(video_names), 11, img_size,img_size, 1))
        for v in range(len(video_names)):
            video = video_names[v]
            if video == 'boats':
                sub_dir = 'boats_big'
                video_dir =os.path.join(videos_path,sub_dir)
            else:
                sub_dir = 'gamma1'
                video_dir =os.path.join(videos_path,sub_dir,video)
            
            if vid_type == 'artificial':
                video_group[v] = ((load_henaff_video(video_dir,natural=False,rgb=rgb,img_size=img_size).permute(0, 2, 3, 1))).clone()
            else:
                video_group[v] = ((load_henaff_video(video_dir,rgb=rgb,img_size=img_size).permute(0, 2, 3, 1))).clone()

        all_videos[vid_type] = video_group.clone()
        # all_videos.append(video_group.copy())
    return all_videos


def load_all_henaff_videos_corrected(videos_path = "/home/gridsan/groups/RosenholtzLab/straightening_models_code/Perceptual-Straightening-models",rgb=True,img_size=512, imagenet=False):
    """
    Returns a dictionary.  Each key is a video sequence type.  Each entry is a numpy array containing all videos of that type
    """
    video_types = ['natural', 'contrast','artificial']
    all_videos = {}
    for vid_type in video_types:
        print(vid_type)
        if vid_type == 'contrast':
            video_names = [ 'water-contrast0.5', 'prairieTer-contrastLog0.1', 'boats_contrastCocktail', 'bees_contrastCocktail', 'walking_contrastCocktail', 'egomotion_contrastCocktail', 'smile-contrastLog0.1', 'walking-contrast0.5', 'bees-contrast0.5', 'walking-contrastLog0.1' ]
            
        else:
            video_names = [ 'water', 'prairieTer', 'boats', 'ice3', 'dogville', 'egomotion', 'walking', 'smile', 'bees', 'leaves-wind', 'carnegie-dam', 'chironomus' ]
        
        
        if rgb:
            video_group = torch.empty((len(video_names), 11, 3, img_size,img_size))
        else:
            video_group = torch.empty((len(video_names), 11, img_size,img_size, 1))
            
        for v,vid in enumerate(video_names):
            if vid_type == 'natural':
                x = (utilities.makeGroundtruth(vid,11,img_size,rgb, imagenet)).clone()
            elif vid_type == 'contrast':
                x = (utilities.makeContrastfade(vid,11,img_size,rgb,imagenet)).clone()
            elif vid_type == 'artificial':
                x = (utilities.makePixelfade(vid,11,img_size,rgb,imagenet)).clone()
                
            if rgb:
                # print(video_group[v].shape, x.shape)
                video_group[v] = x.clone()
            else:
                video_group[v] = x[:,:,:,None].clone()

        all_videos[vid_type] = video_group.clone()
        
    return all_videos

def plot_henaff_video(video, save_name='walking',rgb=False):
    fig, axes = plt.subplots(1,11,figsize=(20,12))
    for i in range(11):
        frame = video[i,:,:]
        frame = frame - torch.min(frame)
        frame = frame / torch.max(frame)
        if rgb:
            axes[i].imshow(torch.permute(frame,(1,2,0)).detach().numpy(), vmin=0, vmax=1)
        else:
            axes[i].imshow(torch.squeeze(frame).detach().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i].grid(False)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.savefig(f'{save_name}.png',dpi=300)
    
def make_gif(frames, duration=200, name='stimulus', directory='.'):
    """
    Make a gif with video frames.  
    Each frame must be a Tensor.  "frames" can be a list of frames or a Tensor
    Uses PIL
    """
    toPIL = torchvision.transforms.ToPILImage()
    gif_frames = []
    for fr in frames:
        if len(fr.size())== 3 and fr.size()[0] > 4:
            try:
                fr = torch.permute(fr,(2,0,1))
            except:
                fr = torch.transpose(fr,0,2)
                fr = torch.transpose(fr,1,2)
        gif_frames.append(toPIL(fr))
    gif_frames[0].save(fp=os.path.join(directory,f'{name}.gif'), format='GIF', append_images=gif_frames,
         save_all=True, duration=duration, loop=0)
    
def collect_model_curves(model_blocks, videos, pca=False, use_outputs=False, vit=False):
    # go through videos and get curvature for different model blocks
    curves = torch.zeros((len(videos), len(model_blocks) + 1))
    curves_full = torch.zeros((len(videos), len(model_blocks) + 1))
    
    
#     if vit:
#         video_curve = torch.mean(computeDistCurv_batch(videos),1)
#         curves_full[:,0] = video_curve 
#         block = model_blocks[b]
#         for b in range(2):
#             model_curve = computeDistCurv_batch(block[b])
#             global_curve = torch.mean(model_curve,axis=1)
#             curves_full[:,b+1] = global_curve
#         curve_change = curves_full - curves_full[:,0:1].repeat(1,curves_full.size(1))
#         return curves_full, curve_change
    
    if use_outputs and not vit:
        try:
            video_curve = torch.mean(computeDistCurv_batch(videos),1)
        except:
            video_curve = torch.mean(computeDistCurv_batch(videos.contiguous()),1)
        curves_full[:,0] = video_curve 
        for b in range(len(model_blocks)):
            block = model_blocks[b]
            try:
                model_curve = computeDistCurv_batch(block)
            except:
                model_curve = computeDistCurv_batch(block.contiguous())
            global_curve = torch.mean(model_curve,axis=1)
            curves_full[:,b+1] = global_curve
        curve_change = curves_full - curves_full[:,0:1].repeat(1,curves_full.size(1))
        return curves_full, curve_change
    
    else:
        if vit:
            curves = torch.zeros((len(videos), 2 + 1))
            curves_full = torch.zeros((len(videos), 2 + 1))
            for i in range(len(videos)):
                print(i)
                # get curve
                model_curve, video_curve = get_model_curve(model_blocks, 
                                                           videos[i], pca=pca, use_outputs=use_outputs,vit=True)

                global_curve = [torch.mean(c).item() for c in model_curve]
                global_curve.insert(0,torch.mean(video_curve).item())
                print(global_curve)

                curves_full[i,:] = torch.FloatTensor(global_curve)

                # get curve change
                curve_change = get_curve_change(model_curve, video_curve)
                curves[i,:] = torch.FloatTensor(curve_change)

            return curves_full, curves
        else:

            for i in range(len(videos)):
                print(i)
                # get curve
                model_curve, video_curve = get_model_curve(model_blocks, 
                                                           videos[i], pca=pca, use_outputs=use_outputs)

                global_curve = [torch.mean(c).item() for c in model_curve]
                global_curve.insert(0,torch.mean(video_curve).item())

                curves_full[i,:] = torch.FloatTensor(global_curve)

                # get curve change
                curve_change = get_curve_change(model_curve, video_curve)
                curves[i,:] = torch.FloatTensor(curve_change)

        return curves_full, curves

def get_global_curve(curves):
    return torch.mean(curves,0)


def calculate_list_similarity(list_a, list_b, method='kendaltau'):
    '''
    Function to calculate similarity of two ordered lists, only supports RBO method for now, but could support other methods in the future
    Parameters:
        list_a (list) first list
        list_b (list) second list
        metric (string) Method of comparing lists
    Returns:
        similarity (float) single float representing list similarity ([0,1] for RBO, [-1,1] for KT)
    '''
    # codebase for RBO here: https://github.com/changyaochen/rbo
    # if method == 'RBO':
    #     print(list_a, list_b)
    #     similarity = 1
    if method == 'kendaltau':
        from scipy import stats
        similarity, p = stats.kendalltau(list_a, list_b)
    else:
        print(f'ERROR: Similarity Metric {metric} Not implemented!')
        
    return(similarity)

def write_model_csv(file_dir, filename, curve_dict):
    header = ['free_params', 'test_acc', 'adversarial_acc', 'training_epoch_time', 'MSE', 'perceptual_loss']
    header_curves = ['layer_names', 'natural_curves','natural_ste', 'contrast_curves','contrast_ste','artificial_curves','artificial_ste']
    full_header = header + header_curves
    num_layers = len(curve_dict['layer_names'])
    data = (np.ones((num_layers,len(full_header)),dtype=object)*float('NaN'))
    for i in range(len(header_curves)):
        if header_curves[i] == 'layer_names':
            data[:,len(header)+i] = curve_dict[header_curves[i]]
        else:
            data[:,len(header)+i] = curve_dict[header_curves[i]].tolist()


    with open(os.path.join(file_dir,f'{filename}.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(full_header)

        # write the data
        for r in range(num_layers):
            writer.writerow(data[r,:])