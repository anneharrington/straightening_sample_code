import numpy as np
import torch
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from prednet_kitti_utils import SequenceGenerator

import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import tensorflow as tf
import math


def computeDistCurv_tf(x):
    x = tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1],-1])
    v = x[:,1:] - x[:,0:-1]
    d = tf.math.sqrt(tf.math.reduce_sum(v**2,axis=2))
    v = tf.math.divide(v, tf.expand_dims(d,axis=2 ))
    c = tf.math.reduce_sum(v[:,1:]*v[:,0:-1], axis=2)
    c = tf.math.acos(tf.clip_by_value( c,clip_value_min=-1,clip_value_max=1 ))
    c = tf.math.multiply(c, 180. / math.pi )
    return c 

def load_prednet_weights(weights_path='./prednet_weights/prednet_kitti_weights.hdf5',
                        json_path ='./prednet_weights/prednet_kitti_model.json'):
    # Load trained model
    f = open(json_path, 'r')
    json_string = f.read()
    f.close()
    train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet,'tf': tf, 
                        'math':math,'computeDistCurv_tf':computeDistCurv_tf})
    train_model.load_weights(weights_path)
    return train_model

def get_prednet_blocks(model,block_names=['R3', 'R2', 'R1', 'R0', 'prediction'],
                       time_steps=10, input_shape=None):
    # Create testing model (to output predictions)
    layer_config = model.layers[1].get_config()

    output_modes = block_names

    model_blocks = []

    for mode in output_modes:
        print(mode)
        # make the model sections
        layer_config['output_mode'] = mode
        data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
        test_prednet = PredNet(weights=model.layers[1].get_weights(), **layer_config)
        
        if input_shape is None:
            input_shape = list(model.layers[0].batch_input_shape[1:])
            input_shape[0] = time_steps
            
        inputs = Input(shape=tuple(input_shape))
        predictions = test_prednet(inputs)
        test_model = Model(inputs=inputs, outputs=predictions)

        model_blocks.append(test_model)
        
    return model_blocks

def load_kitti_test(test_path="../prednet_coxlab/kitti_hkl_py3/X_test.hkl", 
                    test_sources="../prednet_coxlab/kitti_hkl_py3/sources_test.hkl",
                    time_steps=10, sequence_start_mode='unique', data_format='channels_last'):
    
    test_generator = SequenceGenerator(test_path, test_sources, time_steps, sequence_start_mode=sequence_start_mode, data_format=data_format)
    X_test = test_generator.create_all()
    if data_format == 'channels_first':
        X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    return X_test

def get_prednet_outputs(input_videos, model_blocks, batch_size=10, omit_first=True, convert_torch=True):
    """
    Given:
        input_videos (must be numpy array size (videos, frames, height, width, channels),
        model_blocks (list of model instances),
        omit_first (bool indicating whether to omit the fist frame of the output,
            use if finding curvature after since the first prediction is in valid)
        convert_torch (bool indicating whether to convert outputs to pytorch tensor,
            use if finding curvature after)
    Returns:
        numpy array of model outputs
    """
    model_outputs = []

    for m in range(len(model_blocks)):
        print(m)
        model = model_blocks[m]
        layer_config = model.layers[1].get_config()
        data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
        X_hat = model.predict(input_videos, batch_size)
        if data_format == 'channels_first':
            X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))
        if convert_torch:
            X_hat  = torch.from_numpy(X_hat)
        if omit_first:
            X_hat = X_hat[:,1:,:,:,:]
        model_outputs.append(X_hat)
        print(X_hat.shape)
        
    return model_outputs

def save_prediction_plots(num_vids, videos, outputs, output_dir, name, nt):
    # plot up to num_vids predicition results
    for i in range(num_vids):
        print(i)

        aspect_ratio = float(outputs[-1].shape[2]) / outputs[-1].shape[3]
        plt.figure(figsize = (nt*2, 8*aspect_ratio))
        gs = gridspec.GridSpec(2, nt)
        gs.update(wspace=0., hspace=0.)


        for t in range(nt):
            plt.subplot(gs[t])
            plt.imshow(videos[i,t], interpolation='none',vmin=0, vmax=1)
            if t==0: plt.ylabel('Actual', fontsize=10)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])


            plt.subplot(gs[t + nt])
            if t!=0:
                plt.imshow(outputs[-1][i,t-1], interpolation='none',vmin=0, vmax=1)
            else:
                plt.imshow(np.zeros(outputs[-1][i,2].shape))
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            if t==0: plt.ylabel('Predicted', fontsize=10)
        plt.savefig(os.path.join(output_dir,f'{name}_prediction_{i}.png'), dpi=300)
        plt.close()
        


if __name__ == "__main__":
    pass