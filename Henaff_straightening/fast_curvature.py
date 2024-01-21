import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils_henaff import *
import torchvision
from torchvision import transforms
import os
import sys

def get_intermediate_activations(model_layers,layer_names=None,vit=False,path=0):
    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            if vit:
                if type(output) == list:
                    activation[name] = output[path].detach()
                else:
                    activation[name] = output.detach()
            else:
                # print(output)
                activation[name] = output.detach()
        return hook
    hooks = []
    for layer in range(len(model_layers)):
        if layer_names is None:
            name = layer
        else:
            name = layer_names[layer]
        hooks.append(model_layers[layer].register_forward_hook(getActivation(name)))

    return hooks, activation

# def get_intermediate_curv_old(model, model_layers,videos, layer_names=None, hen=False, vit=False, path=0):
#     layerwise_curvs = torch.zeros((videos.size(0), len(model_layers)+2))
#     for v in range(videos.size(0)):
#         video = videos[v]

#         #pixel curvature
#         pixel_curv = torch.mean(computeDistCurv(video.detach().cpu())[1]).item()
#         layerwise_curvs[v,0] = pixel_curv

#         #model layer curvature
#         #if this is henaff's bio model that can't batch, do this framewise
#         if(hen):
#             model_output = []
#             for f, frame in enumerate(video):
#                 hooks, activation = get_intermediate_activations(model_layers,layer_names,vit=vit,path=path)
#                 model_output.append(model(frame))
#             output_curv = torch.mean(computeDistCurv(model_output.detach().cpu())[1]).item()
#             layerwise_curvs[v,len(model_layers)+1] = output_curv

#             for h in range(len(model_layers)):
#                 hooks[h].remove()

#             l = 1
#             for key in activation.keys():
#                 try:
#                     layerwise_curvs[f,v,l] = torch.mean(computeDistCurv(activation[key].detach().cpu())[1]).item()
#                 except:
#                     layerwise_curvs[f, v,l] = torch.mean(computeDistCurv(activation[key].detach().cpu().contiguous())[1]).item()
#                 l+=1
#         #otherwise use batching!        
#         else:
#             hooks, activation = get_intermediate_activations(model_layers,layer_names,vit=vit,path=path)
#             model_output = model(video)
#             output_curv = torch.mean(computeDistCurv(model_output.detach().cpu())[1]).item()
#             layerwise_curvs[v,len(model_layers)+1] = output_curv

#             for h in range(len(model_layers)):
#                 hooks[h].remove()

#             l = 1
#             for key in activation.keys():
#                 try:
#                     layerwise_curvs[v,l] = torch.mean(computeDistCurv(activation[key].detach().cpu())[1]).item()
#                 except:
#                     layerwise_curvs[v,l] = torch.mean(computeDistCurv(activation[key].detach().cpu().contiguous())[1]).item()
#                 l+=1

    
#         del hooks, activation
#     return layerwise_curvs


def get_intermediate_curv(model, model_layers,videos, layer_names=None, hen=False, vit=False, path=0,pca=False,num_pcs=10, with_pixel_pc = False):
    layerwise_curvs = torch.zeros((videos.size(0), len(model_layers)+2))
    for v in range(videos.size(0)):
        video = videos[v]

        #pixel curvature
        vid_for_curve = video.detach().cpu()
        if pca and with_pixel_pc:
            vid_for_curve, _, _ = torch_pca_projection(video.detach().cpu(), num_pcs=num_pcs)
            
        pixel_curv = torch.mean(computeDistCurv(vid_for_curve)[1]).item()
        layerwise_curvs[v,0] = pixel_curv

        #model layer curvature
        #if this is henaff's bio model that can't batch, do this framewise
        if(hen):
            model_output = []
            for f, frame in enumerate(video):
                hooks, activation = get_intermediate_activations(model_layers,layer_names,vit=vit,path=path)
                model_output.append(model(frame))
            output_curv = torch.mean(computeDistCurv(model_output.detach().cpu())[1]).item()
            layerwise_curvs[v,len(model_layers)+1] = output_curv

            for h in range(len(model_layers)):
                hooks[h].remove()

            l = 1
            for key in activation.keys():
                try:
                    layerwise_curvs[f,v,l] = torch.mean(computeDistCurv(activation[key].detach().cpu())[1]).item()
                except:
                    layerwise_curvs[f, v,l] = torch.mean(computeDistCurv(activation[key].detach().cpu().contiguous())[1]).item()
                l+=1
        #otherwise use batching!        
        else:
            hooks, activation = get_intermediate_activations(model_layers,layer_names,vit=vit,path=path)
            model_output = model(video)
            output_for_curve = model_output.detach().cpu()
            if pca:
                output_for_curve,_,_ = torch_pca_projection(output_for_curve)
            output_curv = torch.mean(computeDistCurv(output_for_curve)[1]).item()
            layerwise_curvs[v,len(model_layers)+1] = output_curv

            for h in range(len(model_layers)):
                hooks[h].remove()

            l = 1
            for key in activation.keys():
                act = activation[key].detach().cpu().contiguous()
                if pca:
                    act,_,_ = torch_pca_projection(act,num_pcs=num_pcs)
                layerwise_curvs[v,l] = torch.mean(computeDistCurv(act)[1]).item()
                l+=1

    
        del hooks, activation
    return layerwise_curvs

        