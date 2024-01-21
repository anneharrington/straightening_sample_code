import os
import torch
import numpy 
from PIL import Image
import math 
from torchvision import transforms

def makeLoaddir(imgName):

	loaddir = '/home/gridsan/groups/RosenholtzLab/straightening_models_code/Perceptual-Straightening-models/stimuli/'

	if imgName == 'boats':
		loaddir = loaddir + 'boats_big'
	else:
		loaddir = loaddir + 'gamma1/' + imgName 

	loaddir = loaddir + '/groundtruth'

	return loaddir


# def makeGroundtruth(imgName,nsmpl=11,imgSize=256,rgb=False,imagenet=False):

# 	loaddir = makeLoaddir(imgName)
# 	# imgSize = params.imgSize
# 	vid = torch.Tensor( nsmpl, imgSize, imgSize )
# 	for i in range( nsmpl ):
# 		loadfile = loaddir + str(i+1) + '.png'
# 		img = Image.open(loadfile)
# 			# img = Image.open(loadfile).convert('RGB') # this isn't working ...
# 		img = img.resize( (imgSize,imgSize), resample=Image.LANCZOS )       
# 		img = torch.from_numpy(numpy.array(img))
# 		if img.dim() > 2:
# 			img = img.select( 2, 0 )
# 		vid[i].copy_(img)
# 	if rgb:
# 		vid = torch.cat((vid[...,None],vid[...,None],vid[...,None]),dim=3)
# 	return vid 


def makeGroundtruth(imgName,nsmpl=11,imgSize=256,rgb=False,imagenet=False):

	loaddir = makeLoaddir(imgName)
	# imgSize = params.imgSize
	if rgb:
		vid = torch.Tensor(nsmpl,3,imgSize,imgSize)
	else:
		vid = torch.Tensor( nsmpl, imgSize, imgSize )
	for i in range( nsmpl ):
		loadfile = loaddir + str(i+1) + '.png'
		img = Image.open(loadfile)
			# img = Image.open(loadfile).convert('RGB') # this isn't working ...
		img = img.resize( (imgSize,imgSize), resample=Image.LANCZOS )       
		img = torch.from_numpy(numpy.array(img))/255.
		if img.dim() > 2:
			img = img.select( 2, 0 )
		if rgb:
			# create 3-channel image with channels first
			img = torch.cat((img[None,...],img[None,...],img[None,...]),dim=0)
			if imagenet:
				normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
				img = normalize(img)
		vid[i].copy_(img)
	# if rgb:
	# 	vid = torch.cat((vid[...,None],vid[...,None],vid[...,None]),dim=3)
	return vid 

def linearlyInterpolate(x):
	imgA = x[ 0 ].clone()
	imgB = x[-1 ].clone()
	for i in range(x.size(0)):
		t = i / (x.size(0)-1)
		x[i].copy_( imgA ).mul_( 1-t ).add_( t, imgB )



def makePixelfade(imgName,nsmpl=11,imgSize=256,rgb=False,imagenet=False):
	x = makeGroundtruth(imgName,nsmpl,imgSize,rgb,imagenet)
	linearlyInterpolate( x ) 
	return x 

def makeContrastfade(imgName,nsmpl=11,imgSize=256,rgb=False,imagenet=False):

	x = makeGroundtruth(imgName,nsmpl,imgSize,rgb,imagenet)

	if imgName in ['water-contrast0.5', 'walking-contrast0.5', 'bees-contrast0.5', 'boats_contrastCocktail', 'walking_contrastCocktail']:

		midGrey = x[0][0][0]
		x[-1] = 0.5 * (x[0] - midGrey ) + midGrey
		linearlyInterpolate( x ) 

	elif imgName in ['prairieTer-contrastLog0.1', 'walking-contrastLog0.1', 'smile-contrastLog0.1', 'bees_contrastCocktail', 'egomotion_contrastCocktail']:

		midGrey = x[0][0][0]
		x = x - midGrey
		for i in range( 1, x.size(0) ):
			t = i / ( x.size(0) - 1 )
			x[i] = x[0] * 0.1**t
		x = x + midGrey

	else:
		raise Exception('sorry, no data for this sequence!')

	return x 


def computeDistCurv( x ):

	x = x.view( x.size(0), -1 ) 
	v = x[1:] - x[0:-1]
	d = (v**2).sum(1).sqrt()
	v.div_( d.unsqueeze(1) )
	c = ( v[1:] * v[0:-1] ).sum(1).clamp( -1, 1 ).acos()
	c.mul_( 180 / math.pi )

	return d, c