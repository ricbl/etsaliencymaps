#!/usr/bin/env python
# coding: utf-8

import os
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-fs','--folder_to_save', type=str, default=None)
ap.add_argument('-fl','--folder_to_load', type=str, default=None)
ap.add_argument('-g', '--gpus', type=str, default='0')
args = ap.parse_args()
if args.gpus is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
else:
    args.gpus = os.environ['CUDA_VISIBLE_DEVICES']

import torch

import pandas as pd
import numpy as np
from collections import OrderedDict
from PIL import Image
import cv2
from .config_paths import eyetracking_dataset_path
from .dataset import val_transform, pre_process_path
from .get_model import get_model
import imageio

import skimage.transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_to_save = args.folder_to_save
folder_to_load = args.folder_to_load

def crop_or_pad_to(image, final_size):
    assert(len(image.shape)==len(final_size))
    for i in range(len(image.shape)):

        padding_list = [[0,0]]*len(image.shape)
        if (-image.shape[i] + final_size[i])%2 == 0:
            a = (-image.shape[i] + final_size[i])//2
            size_to_pad = [a, a]
        else:
            size_to_pad = [(-image.shape[i] + final_size[i]+1)//2 , (-image.shape[i] + final_size[i]-1)//2]
        padding_list[i] = size_to_pad
        padding_list = np.array(padding_list)
        if image.shape[i]<final_size[i]:
            image = skimage.util.pad(image, padding_list, mode = 'constant', constant_values  = 0.0)
        elif image.shape[i]>final_size[i]:
            image = skimage.util.crop(image,-padding_list,copy = True)
    return image

class HeatmapGenerator():
    def __init__ (self, model, mode):
        self.model = model
        self.model.eval()
        self.mode = mode
    
    def mix_weights_of_classes(self, label_indices, weights, pred_global, imageData, shape):
        heatmaps = []
        for i, label_idx in enumerate(label_indices):
            hm = self.get_heatmap(pred_global, label_idx, imageData, shape)
            hm = hm*weights[i]
            if(np.isnan(hm).all()):
                print(pred_global)
                assert(False)
            heatmaps.append(hm)
        return sum(heatmaps)
    
    def get_global_heatmap(self, imageData, shape, mode):
        pred_global =self.model(imageData)
        if(mode=='max_class_only'):
            label_indices = torch.max(pred_global,1)[1]
            weights = [1]
        elif(mode=='weighted_classes'):
            label_indices = range(len(pred_global[0]))
            weights = torch.sigmoid(pred_global)[0].detach().cpu().numpy()
        elif(mode=='uniform'):
            label_indices = range(len(pred_global[0]))
            weights =[1]*len(pred_global[0])
        elif(mode=='thresholded'):
            label_indices = torch.where(pred_global[0]>=0)[0]
            if(len(label_indices)==0):
                label_indices = [0]
            weights = [1]*len(label_indices)
        return self.mix_weights_of_classes(label_indices, weights, pred_global, imageData, shape)
    
    #--------------------------------------------------------------------------------
    def get_heatmap_index(self, gradients, activations, shape):
        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdims = True)
        map = activations * pooled_gradients
        
        # average the channels of the activations
        heatmap = torch.sum(map, dim=1).squeeze()

        if(torch.all(map==0)):
            print(torch.sum(pooled_gradients))
            print(torch.sum(activations))
            print(activations)
            assert(False)
        npHeatmap = heatmap.cpu().numpy()
        cam = cv2.resize(npHeatmap, (max(shape), max(shape)))
        cam = crop_or_pad_to(cam, shape)
        assert(cam.shape == shape)
        return cam
    
    def get_heatmap(self, pred, label_idx, imageData, shape):
        pred[:, label_idx].backward(retain_graph = True)

        # pull the gradients out of the model
        gradients = self.model.get_activations_gradient()
        # get the activations of the last convolutional layer
        activations = [activation.detach() for activation in self.model.get_activations(imageData)]
        
        cams = []
        nchannels = []
        for i in range(len(gradients)):
            nchannels.append(activations[i].size(1))
            cams.append(self.get_heatmap_index(gradients[i], activations[i], shape))
        
        return np.maximum(sum(cams)/sum(nchannels), 0) 
        

    def generate(self, pathImageFile, trial, phase, model_name, use_gpu=True):
        
        img = imageio.imread(pathImageFile)
        imageData = Image.fromarray(img)
        
        shape = (img.shape[0], img.shape[1])
        
        imageData = val_transform(imageData)
        imageData = imageData.unsqueeze_(0)
        if use_gpu:
            imageData = imageData.to(device)
        cam_np = self.get_global_heatmap(imageData, shape, mode=self.mode)
        info_dict = {'np_image': cam_np, 'img_path': pathImageFile, 'trial': trial}
        import pathlib
        pathlib.Path(f'{folder_to_save}/{self.mode}/{model_name}/phase_{phase}_0.98' + '/').mkdir(parents=True, exist_ok=True) 
        np.save(f'{folder_to_save}/{self.mode}/{model_name}/phase_{phase}_0.98' + '/' + str(trial), info_dict)
        
        im = Image.fromarray((cam_np-np.min(cam_np))/(np.max(cam_np)-np.min(cam_np))*255).convert('RGB')
        im.save(f'{folder_to_save}/{self.mode}/{model_name}/phase_{phase}_0.98' + '/' + str(trial) + '.png')
        
        im = Image.fromarray(imageData[0,0,...].cpu().numpy()*255).convert('RGB')
        im.save(f'{folder_to_save}/{self.mode}/{model_name}/phase_{phase}_0.98' + '/' + str(trial) + 'or.png')
        
        if model_name == 'ag_sononet':
            am1, am2 = self.model.get_attentions()
            am1_np = am1.detach().cpu().numpy()
            am1_np = cv2.resize(am1_np, (max(shape), max(shape)))
            am1_np = crop_or_pad_to(am1_np, shape)
            assert(am1_np.shape == shape)
            info_dict = {'np_image': am1_np, 'img_path': pathImageFile, 'trial': trial}
            import pathlib
            pathlib.Path(f'{folder_to_save}/{self.mode}/{model_name}/phase_{phase}_0.98' + '/').mkdir(parents=True, exist_ok=True) 
            np.save(f'{folder_to_save}/{self.mode}/{model_name}/phase_{phase}_0.98' + '/' + str(trial)+ '_am1', info_dict)
            im = Image.fromarray((am1_np-np.min(am1_np))/(np.max(am1_np)-np.min(am1_np))*255).convert('RGB')
            
            im.save(f'{folder_to_save}/{self.mode}/{model_name}/phase_{phase}_0.98' + '/' + str(trial) + '_am1.png')
            
            am2_np = am2.detach().cpu().numpy()
            am2_np = cv2.resize(am2_np, (max(shape), max(shape)))
            am2_np = crop_or_pad_to(am2_np, shape)
            assert(am2_np.shape == shape)
            info_dict = {'np_image': am2_np, 'img_path': pathImageFile, 'trial': trial}
            import pathlib
            pathlib.Path(f'{folder_to_save}/{self.mode}/{model_name}/phase_{phase}_0.98' + '/').mkdir(parents=True, exist_ok=True) 
            np.save(f'{folder_to_save}/{self.mode}/{model_name}/phase_{phase}_0.98' + '/' + str(trial) + '_am2', info_dict)
            im = Image.fromarray((am2_np-np.min(am2_np))/(np.max(am2_np)-np.min(am2_np))*255).convert('RGB')
            im.save(f'{folder_to_save}/{self.mode}/{model_name}/phase_{phase}_0.98' + '/' + str(trial) + '_am2.png')
            
            assert(am2_np.shape == shape)

# ## Get list of images from phase 1 and phase 2
def get_filepaths(filename_edf):
    
    filepaths = []
    
    df = pd.read_csv(filename_edf)
    df = df[df['eye_tracking_data_discarded']==False]
    filepaths = df['image'].values
    filepaths_trials = sorted(set(filepaths))
    return filepaths_trials

def run_one_phase(phase, edf_file, model_name, h):
    phase_images = get_filepaths(filename_edf = edf_file)
    print(f'Generating heatmaps for phase {phase}....')
    for trial, filepath in enumerate(phase_images):
        print('trial', trial)
        h.generate(pre_process_path(filepath), trial, phase, model_name)
    
# ## Load model
def generate_heatmaps(model_name, method_mixing):
    model = get_model(model_name, None)
    trained_model_path = os.path.join(f'{folder_to_load}/', model_name)
    checkpoint = torch.load(os.path.join(trained_model_path, 'best_model.pt'), map_location = device)
    new_state_dict = OrderedDict()
    for k1, k2 in zip(model.state_dict(), checkpoint['state_dict']):
        new_state_dict[k1] = checkpoint['state_dict'][k2]
    model.load_state_dict(new_state_dict)
    h = HeatmapGenerator(model, method_mixing)

    ## phase 1
    file_phase = 'metadata_phase_1.csv'
    run_one_phase(1, eyetracking_dataset_path+file_phase, model_name, h)
    ## phase 2
    file_phase = 'metadata_phase_2.csv'
    run_one_phase(2, eyetracking_dataset_path + file_phase, model_name, h)

generate_heatmaps('ag_sononet', 'weighted_classes')
generate_heatmaps('sononet', 'weighted_classes')
generate_heatmaps('sononet', 'thresholded')
generate_heatmaps('ag_sononet', 'thresholded')
generate_heatmaps('sononet', 'uniform')
generate_heatmaps('ag_sononet', 'uniform')