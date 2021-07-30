#!/usr/bin/env python
# coding: utf-8

# # Important notes
# The MIMIC dataset paper
# https://arxiv.org/pdf/1901.07042.pdf
# Refer to this link for the explanation of each of the fields in the mentioned csv files:
# https://physionet.org/content/mimic-cxr-jpg/2.0.0/

import pandas as pd
import os
import numpy as np
from mimic_paths import jpg_path
from mimic_paths import mimic_tables_dir as mimic_dir

label_csv = os.path.join(mimic_dir + 'mimic-cxr-2.0.0-chexpert.csv')
label_df = pd.read_csv(label_csv)

split_csv = os.path.join(mimic_dir + 'mimic-cxr-2.0.0-split.csv')
split_df = pd.read_csv(split_csv)

# ### Merge split df and label_df
# To get a single df with label(s), split, image_name for each image
images_df = pd.merge(left=label_df, right=split_df, left_on = ['subject_id', 'study_id'], right_on=['subject_id', 'study_id'], how='inner')

# ## Filtering out only the good images
# Good images are the ones that are providied in the list image_all_paths.txt
def getImgList(image_path, jpg_path):
    image_list_jpg = []
    with open(image_path) as f:
        image_list = f.readlines()
    for path in image_list:
        temp_path = jpg_path + path.split('files')[-1]
        temp_path = temp_path.replace('.dcm', '.jpg')
        image_list_jpg.append(temp_path.strip())
    return image_list_jpg

image_all_paths = 'image_all_paths.txt'
image_all_paths_jpg = getImgList(image_all_paths, jpg_path)

# ### Get a filtered df with only good paths
good_dicom_ids = []

for i in range(len(image_all_paths_jpg)):
    dc_id = image_all_paths_jpg[i].split('.jpg')[0].split('/')[-1]
    good_dicom_ids.append(dc_id)
images_df_filtered = images_df[images_df['dicom_id'].isin(good_dicom_ids)]
# ### Attaching the path for each good image
path_df = pd.DataFrame({'path': image_all_paths_jpg, 'dicom_id' : good_dicom_ids})
final_df = pd.merge(left=images_df_filtered, right=path_df, on='dicom_id')

# ### Rearanging columns
cols = final_df.columns.to_list()
cols = cols[:2] + cols[-3:] + cols[2:-3]
final_df = final_df[cols]
final_df.head()

# ### Split final_df into Train, Validation and Test dfs
train_df = final_df[final_df['split'] == 'train']
train_df = train_df.drop('split', axis=1)

val_df = final_df[final_df['split'] == 'validate']
val_df = val_df.drop('split', axis=1)

test_df = final_df[final_df['split'] == 'test']
test_df = test_df.drop('split', axis=1)


# ### Getting the list of images which are shown to radiologists in Phase 1 and Phase 2

# Phase 1
collected_data_phase_1_path = 'dataset/metadata_phase_1.csv'

phase_1_list = pd.read_csv(collected_data_phase_1_path)['subject_id'].unique()

# Phase 2
collected_data_phase_2_path = 'dataset/metadata_phase_2.csv'

phase_2_list = pd.read_csv(collected_data_phase_2_path)['subject_id'].unique()

filter_list_phase_1_2 = phase_1_list.tolist() + phase_2_list.tolist()

filter_list_phase_1_2 = pd.DataFrame(np.unique(filter_list_phase_1_2), columns = ['subject_id'])

# ### Filtering out the phase 1 and phase 2 images
# Get subjects to be moved to test_df from train and val set
filter_list_phase_1_2_joined_subjects = final_df[final_df['subject_id'].isin(pd.merge(filter_list_phase_1_2, final_df)['subject_id'])]['path'].values
move_to_test_df = train_df[train_df['path'].isin(filter_list_phase_1_2_joined_subjects)]
move_to_test_df = move_to_test_df.append(val_df[val_df['path'].isin(filter_list_phase_1_2_joined_subjects)])

train_df = train_df[~train_df['path'].isin(filter_list_phase_1_2_joined_subjects)].reset_index(drop=True)
val_df = val_df[~val_df['path'].isin(filter_list_phase_1_2_joined_subjects)].reset_index(drop=True)
test_df = test_df.append(move_to_test_df).reset_index(drop=True)

train_df.to_csv('train_df.csv', index=False)
val_df.to_csv('val_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)

