"""Auxiliary dataset functions
This module provides generic functions related to all datasets
"""

from torch.utils.data import Dataset
import numpy as np
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import os
import h5py

import signal
import hashlib
import PIL
import shutil
import torchvision.transforms as transforms

class MyKeyboardInterruptionException(Exception):
    "Keyboard Interrupt activate signal handler"
    pass
    
def interupt_handler(signum, frame):
    raise MyKeyboardInterruptionException

signal.signal(signal.SIGINT, interupt_handler)

#dataset wrapper to apply transformations to a pytorch dataset. i defines the index of the element
# of the tuple returned by original_dataset to which the transformation should be applied
class TransformsDataset(Dataset):
    def __init__(self, original_dataset, transform, i=0):
        super().__init__()
        self.original_dataset = original_dataset
        self.transform = transform
        if type(self.transform)==type([]):
            self.transform  = transforms.Compose(self.transform)
        self.i = i
    
    def apply_transform_ith_element(self, batch, transform):        
        to_return = *batch[:self.i], transform(batch[self.i]), *batch[(self.i+1):]
        return to_return
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        return self.apply_transform_ith_element(self.original_dataset[index], self.transform)

#generic class to save a pytorch dataset to H5 files, and load them to memory if
# they have already been saved. It assumes that filename is a uniue identifier for the dataset content, 
# such that if filename exists, it will directly load the h5 file and not use original_dataset.
# it also saves a pickle file to store the original organization of the dataset, while the h5
# file store the data.
class H5Dataset(Dataset):
    def __init__(self, original_dataset, path = '.', filename = None):
        super().__init__()
        self.len_ = len(original_dataset)
        if filename is None:
            # if filename is not provided, try to get a hash key for the dataset to characterize its content
            # for several datasets, this will take a really long time, since it has to iterate through the full dataset.
            # it is better to provide your own unique name
            def hash_example(name_, structure, fixed_args):
                structure = np.array(structure)
                structure.flags.writeable = False
                fixed_args['sha1'].update(structure.data)
            sha1 = hashlib.sha1()
            for example in original_dataset:
                apply_function_to_nested_iterators(example, {'sha1': sha1}, hash_example)
            filename = str(sha1.hexdigest())
        filename = filename + self.get_extension()
        self.filepath_h5 = path + '/' + filename
        structure_file = path + '/' + filename + '_structure.pkl'
        if not os.path.exists(self.filepath_h5) or not os.path.exists(structure_file):
            try:
                with self.get_file_handle()(self.filepath_h5, 'w') as h5f:
                    structure = self.create_h5_structure([item for index,item in enumerate(original_dataset) if index<len(original_dataset)], h5f, 1)
                    # for index in range(len(original_dataset)): 
                        # self.pack_h5(original_dataset[index], index, h5f)
                with open(structure_file, 'wb') as output:
                    pickle.dump(structure, output, pickle.HIGHEST_PROTOCOL)
            except Exception as err:
                # if there is an error in the middle of writing, delete the generated files
                # to not have corrupted files
                if os.path.exists(self.filepath_h5):
                    if not os.path.isdir(self.filepath_h5):
                        os.remove(self.filepath_h5)
                    else:
                        shutil.rmtree(self.filepath_h5)
                if os.path.exists(structure_file):
                    if not os.path.isdir(structure_file):
                        os.remove(structure_file)
                    else:
                        shutil.rmtree(structure_file)
                raise Exception('Error while writing hash ' + filename + '. Deleting files ' + self.filepath_h5 + ' and ' + structure_file).with_traceback(err.__traceback__)
        self.file = None
        with open(structure_file, 'rb') as input:
            self.structure = pickle.load(input)
    
    def get_file_handle(self):
        return h5py.File
        
    def get_extension(self):
        return '.h5'
        
    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.filepath_h5, 'r', swmr = True)
        return self.unpack_h5(self.structure, index, self.file)
    
    def __len__(self):
        return self.len_
        
    @staticmethod
    def create_h5_structure(structure, h5f, n_images):
        def function_(name_, value, fixed_args):
            if fixed_args['h5f'] is not None:
                fixed_args['h5f'].create_dataset(name_, data = np.array(value))
            return None
        return apply_function_to_nested_iterators(structure, {'n_images':n_images, 'h5f': h5f}, function_)
    
    @staticmethod
    def unpack_h5(structure, index, h5f):
        return apply_function_to_nested_iterators(structure, {'index':index, 'h5f': h5f}, lambda name_, value, fixed_args: fixed_args['h5f'][name_][:])

#auxiliary function to iterate and apply functions to all elements of a variable composed
# of nested variable of these types: list, tuple, dict
# leafs have to be of kind: np.ndarray, int, float, bool, PIL.Image.Image
def apply_function_to_nested_iterators(structure, fixed_args, function_, name_ = "root"):
    if structure is None or isinstance(structure, (int, float, bool, PIL.Image.Image, np.float32, np.ndarray)):
        return function_(name_, structure, fixed_args)
    elif isinstance(structure, (list, tuple)) or (isinstance(structure, np.ndarray) and 'batch_nparray_open' in fixed_args.keys() and not fixed_args['batch_nparray_open']):
        if isinstance(structure, np.ndarray):
            fixed_args['batch_nparray_open'] = True
        if 'index' in fixed_args.keys() and fixed_args['index'] is not None:
            fixed_args_index = fixed_args['index']
            fixed_args['index'] = None
            return [apply_function_to_nested_iterators(item, fixed_args, function_, name_ = name_ + "/" + '_index_' + str(fixed_args_index) + "/" + '_index_' + str(index)) for index, item in enumerate(structure[fixed_args_index])]
        return [apply_function_to_nested_iterators(item, fixed_args, function_, name_ = name_ + "/" + '_index_' + str(index)) for index, item in enumerate(structure)]
    elif isinstance(structure, dict):
        return {key: apply_function_to_nested_iterators(item, fixed_args, function_, name_ = name_ + "/" + key) for key, item in structure.items()}
    else:
        raise ValueError('Unsuported type: ' + str(type(structure)))