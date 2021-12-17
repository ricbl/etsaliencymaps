import torch
from torch.utils.data import Dataset
import numpy as np
import imageio
from PIL import Image
from .utils_dataset import TransformsDataset, H5Dataset, LoadToMemory
import pandas as pd
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import math
from .config_paths import jpg_path, h5_path

num_workers = 12

str_labels = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Lesion',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices']

def get_auc(y, y_pred):
    auc_list = []
    for i in range(len(str_labels)):
        auc = roc_auc_score(y.detach().cpu()[:,i], y_pred.detach().cpu()[:,i])
        auc_list.append(auc)
    return auc_list

def pre_process_path(dicom_path):
    temp_path = jpg_path + '/physionet.org/files/' + dicom_path.split('files')[-1]
    temp_path = temp_path.replace('.dcm', '.jpg')
    return temp_path.strip()

def get_train_val_dfs():
    train_df = pd.read_csv('./train_df.csv')
    val_df = pd.read_csv('./val_df.csv')
    test_df = pd.read_csv('./test_df.csv')
    return train_df, val_df, test_df

class XRayResizerAR(object):
    def __init__(self, size, fn):
        self.size = size
        self.fn = fn
    
    def __call__(self, img):
        old_size = img.shape[:2]
        ratio = float(self.size)/self.fn(old_size)
        new_size = tuple([round(x*ratio) for x in old_size])
        a = ToImage()(img)
        a = transforms.Resize(new_size)(a)
        return a

def get_32_size(shape, size):
    projected_max_size = size/min(np.array(shape))*max(np.array(shape))
    return round(projected_max_size/16)*16
    

class XRayResizerPadRound32(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        self.resizer = XRayResizerAR(get_32_size(img.shape[:2], self.size), max)
        img = self.resizer(img)
        img = ToNumpy()(img)
        pad_width = (-np.array(img.shape[:2])+max(np.array(img.shape[:2])))/2
        return np.pad(img, ((math.floor(pad_width[0]),math.ceil(pad_width[0])),(math.floor(pad_width[1]),math.ceil(pad_width[1]))))

class ToImage(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return Image.fromarray(tensor)

class ToNumpy(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return np.array(tensor)

class GrayToThree(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return np.tile(tensor[:, :, None], [1,1,3])
        
class MIMICCXRDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.dataset_size = len(self.df)
        self.labels = (self.df[str_labels].astype('float').values > 0) * 1.

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        filepath = self.df.iloc[idx]["path"]
        img = imageio.imread(filepath)
        img = Image.fromarray(img)
        return img, np.array(self.labels[idx]).astype(np.float32)

    def __len__(self):
        return self.dataset_size

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

pre_transform_val = [
                        ToNumpy(),
                        XRayResizerPadRound32(224),
                    ]
post_transform_val = [ 
                        GrayToThree(),
                        transforms.ToTensor(), 
                        normalize
                    ]
val_transform = transforms.Compose(pre_transform_val +  [ToNumpy()] + post_transform_val)

pre_transform_train = [transforms.Resize(224)]
post_transform_train = [
                        GrayToThree(),
                        ToImage(),
                        transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                         scale=(0.95, 1.05), fill=128),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(),            
                        transforms.ToTensor(), 
                        normalize
                        ]

def get_dataset(val_split):
    train_df, val_df, test_df = get_train_val_dfs()
    if val_split == 'test':
        valset = TransformsDataset(H5Dataset(lambda: LoadToMemory(
                TransformsDataset(MIMICCXRDataset(test_df), pre_transform_val)
            ,parallel=True),path = h5_path, filename = 'test_dataset', individual_datasets = True),post_transform_val)
    if val_split == 'val':
        valset = TransformsDataset(H5Dataset(lambda: LoadToMemory(
                TransformsDataset(MIMICCXRDataset(val_df), pre_transform_val)
            ,parallel=True),path = h5_path, filename = 'val_dataset', individual_datasets = True),post_transform_val)
    
    trainset = TransformsDataset(H5Dataset(lambda: LoadToMemory(
            TransformsDataset(MIMICCXRDataset(train_df), pre_transform_train)
        ,parallel=True),path = h5_path, filename = 'train_dataset', individual_datasets = True),post_transform_train)#TEMP:opt
    
    train_loader = torch.utils.data.DataLoader(trainset, num_workers=num_workers, shuffle=True,
                                          batch_size=64, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(valset, num_workers=num_workers, shuffle=True,
                                    batch_size=1, pin_memory=True)
    return train_loader, val_loader