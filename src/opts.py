"""User configuration file

File organizing all configurations that may be set by user when running the 
train.py script. 
Call python -m src.train --help for a complete and formatted list of available user options.
"""

import argparse
import time
from random import randint
import os
import socket
import glob
import shutil
import sys
import pathlib

def get_opt():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_name', type=str, default='sononet')
    ap.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    ap.add_argument('-ep', '--epochs', type=int, default=75)
    ap.add_argument('-bs', '--batch_size', type=int, default=64)
    ap.add_argument('-is', '--img_size', type=int, default=224)
    ap.add_argument('-g', '--gpus', type=str, default='0')
    ap.add_argument('-ex', '--experiment', type=str, default='')
    ap.add_argument('-l', '--load_checkpoint_d', type=str, default=None)
    ap.add_argument('-v', '--val_split', type=str, default='val')
    
                                    
                                    
    args = ap.parse_args()
    
    #gets the current time of the run, and adds a four digit number for getting
    #different folder name for experiments run at the exact same time.
    timestamp = time.strftime("%Y%m%d-%H%M%S") + '-' + str(randint(1000,9999))
    args.timestamp = timestamp
    
    #register a few values that might be important for reproducibility
    args.screen_name = os.getenv('STY')
    args.hostname = socket.gethostname()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    else:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES']
    import platform
    args.python_version = platform.python_version()
    import torch
    args.pytorch_version = torch.__version__ 
    import torchvision
    args.torchvision_version = torchvision.__version__
    import numpy as np
    args.numpy_version = np.__version__
    args.save_folder = './runs/'
    pathlib.Path(args.save_folder).mkdir(parents=True, exist_ok=True) 
    args.output_folder = args.save_folder+'/'+args.experiment+'_'+args.timestamp
    os.mkdir(args.output_folder)
    log_configs(args)
    args = vars(args)
    return args

def log_configs(opt):
    with open(opt.output_folder + '/opts.txt', 'w') as f:
        for key, value in sorted(vars(opt).items()):
            f.write(key + ': ' + str(value).replace('\n', ' ').replace('\r', '') + '\r\n')
    save_run_state(opt)
def save_command(opt):
    command = ' '.join(sys.argv)
    with open(f"{opt.output_folder}/command.txt", "w") as text_file:
        text_file.write(command)

def save_run_state(opt):
    if not os.path.exists(f'{opt.output_folder}/src/'):
        os.mkdir(f'{opt.output_folder}/src/')
    [shutil.copy(filename, (f'{opt.output_folder}/src/')) for filename in glob.glob(os.path.dirname(__file__) + '/*.py')]
    save_command(opt)