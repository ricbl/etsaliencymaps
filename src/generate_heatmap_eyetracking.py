#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import csv
import pathlib
from .config_paths import jpg_path, eyetracking_dataset_path
from joblib import Parallel, delayed

def get_gaussian(y,x,sy,sx, sizey,sizex, shown_rects_image_space):
    mu = [y-shown_rects_image_space[1],x-shown_rects_image_space[0]]
    sig = [sy**2,sx**2]
    x = np.arange(0, shown_rects_image_space[2]-shown_rects_image_space[0], 1)
    y = np.arange(0, shown_rects_image_space[3]-shown_rects_image_space[1], 1)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 1] = X
    pos[:, :, 0] = Y
    to_return = np.zeros([sizey,sizex])
    to_return[shown_rects_image_space[1]:shown_rects_image_space[3], shown_rects_image_space[0]:shown_rects_image_space[2]] = multivariate_normal(mu, sig).pdf(pos)
    return to_return

def create_heatmap(sequence_table, size_x, size_y):
    img = np.zeros((size_y, size_x), dtype=np.float32)
    for index, row in sequence_table.iterrows():
        angle_circle = 1
        shown_rects_image_space = [round(row['xmin_shown_from_image']) ,round(row['ymin_shown_from_image']),round(row['xmax_shown_from_image']),round(row['ymax_shown_from_image'])]
        gaussian = get_gaussian(row['y_position'],row['x_position'], row['angular_resolution_y_pixels_per_degree']*angle_circle, row['angular_resolution_x_pixels_per_degree']*angle_circle, size_y,size_x, shown_rects_image_space)
        img += gaussian*(row['timestamp_end_fixation']-row['timestamp_start_fixation'])
    return img/np.sum(img)

def generate_et_heatmaps_for_one_image(trial, image_name,df_this_trial,data_folder,folder_name):
    index_image = 0
    for _, df_this_index_this_trial in df_this_trial.iterrows():
        
        print('trial', trial, 'index_image', index_image)
        image_size_x = int(float(df_this_index_this_trial['image_size_x']))
        image_size_y = int(float(df_this_index_this_trial['image_size_y']))
        fixations = pd.read_csv(f'{data_folder}/{df_this_index_this_trial["id"]}/fixations.csv')
        this_img = create_heatmap(fixations,image_size_x, image_size_y)

        # Saving the array in npy format
        info_dict = {'np_image': this_img, 'img_path': pre_process_path(image_name), 
                    'trial': trial, 'id':df_this_index_this_trial["id"]}
        np.save(folder_name + '/' + str(trial) + '_' + str(index_image), info_dict)
        
        # Log the progress
        with open('logs.csv', 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([trial, folder_name])
        index_image += 1

def create_heatmaps(data_folder, filename_phase, folder_name='heatmaps'):
    
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True) 
    df = pd.read_csv(data_folder + filename_phase)
    df = df[df['eye_tracking_data_discarded']==False]
    all_images = df['image'].unique()
    Parallel(n_jobs=32)(delayed(generate_et_heatmaps_for_one_image)(trial, image_name,df[df['image']==image_name],data_folder,folder_name) for trial, image_name in enumerate(sorted(all_images)))

def pre_process_path(dicom_path):
    temp_path = jpg_path + '/physionet.org/files/' + dicom_path.split('files')[-1]
    temp_path = temp_path.replace('.dcm', '.jpg')
    return temp_path.strip()

if __name__=='__main__':
    # # ### Phase 1
    print('Starting Phase 1...')
    file_phase_1 = 'metadata_phase_1.csv'

    create_heatmaps(eyetracking_dataset_path ,file_phase_1, 
                    folder_name='heatmaps_phase_1_1')

    # ### Phase 2
    print('Starting Phase 2...')
    file_phase_2 = 'metadata_phase_2.csv'

    create_heatmaps(eyetracking_dataset_path, file_phase_2,
                          folder_name='heatmaps_phase_2_1')