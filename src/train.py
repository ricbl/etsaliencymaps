#!/usr/bin/env python
# coding: utf-8

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import os
import csv
import random
import time
from .dataset import get_auc
from .dataset import str_labels as class_names
from .opts import get_opt
args = get_opt()
# ## Fixing the random seed

SEED = random.randrange(2**32 - 1)
print(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ## Setting the default Parameters

MODEL_NAME = args['model_name']
LR = args['learning_rate']
N_EPOCHS = args['epochs']
load_checkpoint_d = args['load_checkpoint_d']
val_split = args['val_split']

# ## Define the device
device = torch.device("cuda:0")

# ## Import the dataset
from .dataset import get_dataset

from .get_model import get_model
        
# ## Trainer Class
class Trainer():
    def __init__(self, model_name=MODEL_NAME, lr=LR, n_epochs=N_EPOCHS, load_checkpoint_d = load_checkpoint_d):
        self.skip_train =  load_checkpoint_d is not None
        self.model = get_model(model_name, load_checkpoint_d)
        self.lr = lr
        self.n_epochs = n_epochs
        
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.optimizer = optim.Adam(params = self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience = 3)
        self.model_name = model_name
        self.model_save_dir = os.path.join(args['output_folder'], self.model_name)
        
        self.train_loader = None
        self.val_loader = None

        self.start_epoch = 0
        self.best_valid_loss = np.inf
    
    def save_checkpoint(self, epoch, best_valid_loss):
        
        torch.save(
                    {
                        'best_epoch': epoch + 1, 
                        'state_dict': self.model.state_dict(), 
                        'best_loss': best_valid_loss, 
                        'optimizer' : self.optimizer.state_dict()
                    }, 
                    
                    os.path.join(self.model_save_dir, 'best_model.pt')
                )
    
    def train(self, iterator):
        self.model.train()
        running_loss = 0
        for i, batch in enumerate(iterator):
            if i%20==0:
                print('Train: {}/{}'.format(i, len(iterator)), end='                             \r')
            
            inputs, labels = batch[0].to(device), batch[1].to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(iterator)
        return epoch_loss
    
    def evaluate(self, iterator):
        """
        The eval function
        """
        
        self.model.eval()

        running_loss = 0
        all_labels = []
        all_outputs = []

        with torch.no_grad():   
            for i, batch in enumerate(iterator):
                print('Eval: {}/{}'.format(i, len(iterator)), end='                             \r')
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                all_labels.append(labels)
                all_outputs.append(outputs)
#                 break
            epoch_loss = running_loss / len(iterator)
            
            all_labels = torch.cat(all_labels, dim=0)
            all_outputs = torch.cat(all_outputs, dim=0)
            auc_list = get_auc(all_labels, all_outputs)
        return epoch_loss, auc_list
    
    def epoch_time(self, start_time, end_time):
        """
        The utility function to measure the time taken by an epoch to run
        """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def training(self):
        """
        The training function which does the training by calling train and eval functions
        """
    
        best_valid_loss = self.best_valid_loss
        
        self.train_loader, self.val_loader = get_dataset(val_split)
        
        print('Starting the training...')
        print('*'*100)
        
        # Create the model save dir if it already doesn't exist
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        for epoch in range(self.start_epoch, self.n_epochs):

            print(f'Epoch: {epoch+1:02}')

            start_time = time.time()
            if not self.skip_train:
                train_loss = self.train(self.train_loader)
            else:
                train_loss = 0
            valid_loss, val_auc_list = self.evaluate(self.val_loader)

            epoch_mins, epoch_secs = self.epoch_time(start_time, time.time())

            if not self.skip_train:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.save_checkpoint(epoch, best_valid_loss)
                self.scheduler.step(valid_loss)
            # Log training and validation losses for plotting later
            with open(os.path.join(self.model_save_dir, 'logs.csv'), 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                if (epoch == 0):
                    logwriter.writerow(["epoch", "train_loss", "val_loss","Epoch_min", "Epoch_sec", "Seed","LR"] + class_names)
                logwriter.writerow([epoch+1, train_loss, valid_loss, epoch_mins, epoch_secs, SEED, self.optimizer.param_groups[0]['lr']] + val_auc_list)
            
            print(f'Time: {epoch_mins}m {epoch_secs}s') 
            print(f'Train Loss: {train_loss:.3f}')
            print(f'Val   Loss: {valid_loss:.3f}')
            print('-'*60)
            
        print('The best validation loss is', best_valid_loss)
        print('*'*100)

if __name__ == "__main__":
    # Start the training
    trainer = Trainer()
    print('*'*100)
    print('Training {} model with parameters: {}'.format(MODEL_NAME, args))
    print('*'*100)
    print('Using device', device)
    print('*'*100)
    trainer.training()