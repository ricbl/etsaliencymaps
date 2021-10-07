import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from skimage import morphology, exposure
import torch.nn as nn
import torch
from keras.models import load_model
import numpy as np
import pandas as pd
from pydicom import dcmread
import torchvision
import skimage
import math
from skimage.segmentation import flood_fill
import os
from .config_paths import dicom_path, eyetracking_dataset_path, segmentation_model_path
import pathlib

class UNet(nn.Module):
    def __init__(self, num_channels=1):
        super(UNet, self).__init__()
        num_feat = [64, 128, 256, 512, 1024]

        self.down1 = nn.Sequential(Conv3x3(num_channels, num_feat[0]))

        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[0], num_feat[1]))

        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[1], num_feat[2]))

        self.down4 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[2], num_feat[3]))

        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    Conv3x3(num_feat[3], num_feat[4]))

        self.up1 = UpConcat(num_feat[4], num_feat[3])
        self.upconv1 = Conv3x3(num_feat[4], num_feat[3])

        self.up2 = UpConcat(num_feat[3], num_feat[2])
        self.upconv2 = Conv3x3(num_feat[3], num_feat[2])

        self.up3 = UpConcat(num_feat[2], num_feat[1])
        self.upconv3 = Conv3x3(num_feat[2], num_feat[1])

        self.up4 = UpConcat(num_feat[1], num_feat[0])
        self.upconv4 = Conv3x3(num_feat[1], num_feat[0])

        self.final = nn.Sequential(nn.Conv2d(num_feat[0],
                                             1,
                                             kernel_size=1))

    def forward(self, inputs, return_features=False):
        down1_feat = self.down1(inputs)
        down2_feat = self.down2(down1_feat)
        down3_feat = self.down3(down2_feat)
        down4_feat = self.down4(down3_feat)
        bottom_feat = self.bottom(down4_feat)

        up1_feat = self.up1(bottom_feat, down4_feat)
        up1_feat = self.upconv1(up1_feat)
        up2_feat = self.up2(up1_feat, down3_feat)
        up2_feat = self.upconv2(up2_feat)
        up3_feat = self.up3(up2_feat, down2_feat)
        up3_feat = self.upconv3(up3_feat)
        up4_feat = self.up4(up3_feat, down1_feat)
        up4_feat = self.upconv4(up4_feat)

        if return_features:
            outputs = up4_feat
        else:
            outputs = self.final(up4_feat)

        return outputs

class Conv3x3(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        outputs = self.deconv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out

def find_nearest(a, n):
    return np.argmin(np.abs(a.astype(float)-n))

def apply_windowing(x,level,width):
    return np.minimum(np.maximum(((x.astype(float)-level)/width+0.5),0),1);

def open_dicom(filpath_image_this_trial):
    with dcmread(filpath_image_this_trial) as header:
        max_possible_value = (2**float(header.BitsStored)-1)
        x = header.pixel_array
        x = x.astype(float)/max_possible_value
        if 'WindowWidth' in header:
            if hasattr(header.WindowWidth, '__getitem__'):
                wwidth = header.WindowWidth[0]
            else:
                wwidth = header.WindowWidth
            if hasattr(header.WindowCenter, '__getitem__'):
                wcenter = header.WindowCenter[0]
            else:
                wcenter = header.WindowCenter
            windowing_width = wwidth/max_possible_value
            windowing_level = wcenter/max_possible_value
            if header.PhotometricInterpretation=='MONOCHROME1' or not ('PixelIntensityRelationshipSign' in header) or header.PixelIntensityRelationshipSign==1:
                x = 1-x
                windowing_level = 1 - windowing_level
        else:
             if 'VOILUTSequence' in header:
                lut_center = float(header.VOILUTSequence[0].LUTDescriptor[0])/2
                window_center = find_nearest(np.array(header.VOILUTSequence[0].LUTData), lut_center)
                deltas = []
                for i in range(10,31):
                    deltas.append((float(header.VOILUTSequence[0].LUTData[window_center+i]) - float(header.VOILUTSequence[0].LUTData[window_center-i]))/2/i)
                window_width = lut_center/sum(deltas)*2*len(deltas)
                windowing_width = window_width/max_possible_value
                windowing_level = (window_center-1)/max_possible_value
                if windowing_width < 0:
                    windowing_width = -windowing_width
                    x = 1-x
                    windowing_level = 1 - windowing_level
             else:
                windowing_width = 1
                windowing_level = 0.5;
                if header.PhotometricInterpretation=='MONOCHROME1' or not ('PixelIntensityRelationshipSign' in header) or header.PixelIntensityRelationshipSign==1:
                    x = 1-x
                    windowing_level = 1 - windowing_level
        return apply_windowing(x, windowing_level, windowing_width)

class XRayResizerAR(object):
    def __init__(self, size, fn):
        self.size = size
        self.fn = fn
    
    def __call__(self, img):
        old_size = img.shape[1:]
        ratio = float(self.size)/self.fn(old_size)
        new_size = tuple([round(x*ratio) for x in old_size])
        return skimage.transform.resize(img, (1, new_size[0], new_size[1]), mode='constant', preserve_range=True).astype(np.float32)

class XRayResizerPad(object):
    def __init__(self, size, fn):
        self.resizer = XRayResizerAR(size, fn)
    
    def __call__(self, img):
        img = self.resizer(img)
        pad_width = (-np.array(img.shape[1:])+max(np.array(img.shape[1:])))/2
        return np.pad(img, ((0,0),(math.ceil(pad_width[0]),math.floor(pad_width[0])),(math.ceil(pad_width[1]),math.floor(pad_width[1]))))

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_dataset):
        results_csv = pd.read_csv(metadata_dataset).sort_values(['image'])
        results_csv = results_csv[results_csv['eye_tracking_data_discarded']==False]
        self.dicoms = results_csv['image'].unique()
        self.length = len(self.dicoms)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        x = open_dicom(dicom_path + '/physionet.org/' + self.dicoms[index].split('physionet.org')[-1])
        original_size = x.shape
        x = XRayResizerPad(256, max)(x[None,...])[0,...]
        return x, np.array([ord(c) for c in self.dicoms[index].split('/')[-1][:-4]]),np.array(original_size)

def main(phase):
    dataloader = torch.utils.data.DataLoader(SegmentationDataset(f'{eyetracking_dataset_path}/metadata_phase_{phase}.csv'))
    segmentation_model = SegmentationNetwork()
    pathlib.Path('segmentations_convex').mkdir(parents=True, exist_ok=True) 
    for x, id, original_size in dataloader:
        segmentation = segmentation_model(x)
        crop_size = (segmentation[0,0,:,:].size()-original_size[0].numpy()/max(original_size[0].numpy())*256)/2
        segmentation = segmentation[:,:,math.ceil(crop_size[0]):segmentation.shape[2]-math.floor(crop_size[0]), math.ceil(crop_size[1]):segmentation.shape[3]-math.floor(crop_size[1])]
        segmentation =torch.nn.functional.interpolate(segmentation, size=[original_size[0][0].item(),original_size[0][1].item()])
        torchvision.utils.save_image( segmentation, './segmentations_convex/segmentation_'+''.join([chr(c) for c in id[0]])+'.png')

from skimage.morphology import convex_hull_image
def concave(img):
    init_ls = convex_hull_image(img)*1.0
    return init_ls

class SegmentationNetwork(nn.Module):
    def __init__(self):
        super(SegmentationNetwork, self).__init__()
        self.segmentation = self.get_segmentation
        model_name = f'{segmentation_model_path}/trained_model.hdf5'    
        self.unet = load_model(model_name)
        self.UNet = lambda x: self.unet.predict(x)
    
    def get_segmentation(self, input_to_generator):
        input_to_generator = 1-torch.tensor(exposure.equalize_hist(input_to_generator.cpu().numpy()[0,0,:,:]), dtype = torch.float).cuda().unsqueeze(0).unsqueeze(0)
        input_to_generator = (input_to_generator-(1-0.5489323))/0.23906362
        pred = self.UNet(np.expand_dims(input_to_generator.squeeze(0).cpu().numpy(), 3))[0,..., 0]
        pr = pred > 0.5
        
        pr = self.remove_small_regions(pr, 0.02 * np.prod(pred.shape))
        pr = self.remove_regions_border(pr)
        pr = concave(pr*1.0)
        
        return torch.tensor(pr).unsqueeze(0).unsqueeze(0)
    
    def remove_regions_border(self, img):
        for p in range(img.shape[0]):
            for seed in ((p,0), (p,img.shape[1]-1)):
                if img[seed[0],seed[1]]==1:
                    img = flood_fill(img, seed, 0)
        for p in range(img.shape[1]):
            for seed in ((0,p), (img.shape[0]-1, p)):
                if img[seed[0], seed[1]]==1:
                    img = flood_fill(img, seed, 0)
        return img
        
    
    def remove_small_regions(self, img, size):
        img = morphology.remove_small_objects(img, size)
        img = morphology.remove_small_holes(img, size)
        return img
    
    def get_segmentation_and_features(self, xi_to_segment):
        segmented_xi = self.segmentation(torch.mean(xi_to_segment, dim = 1, keepdim = True))
        segmented_xi = segmented_xi.detach()
        return segmented_xi
    
    def forward(self, xi):
        xi_to_segment = xi.unsqueeze(0)
        xi_to_segment = -xi_to_segment
        segmented_xi = self.get_segmentation_and_features(xi_to_segment)
        return segmented_xi

# main(1)
main(2)