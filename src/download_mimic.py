import pathlib
import pandas as pd
from config_paths import eyetracking_dataset_path
import argparse
import os
from config_paths import jpg_path, dicom_path, mimic_tables_dir
import gzip
import shutil

def get_list_images_dicom(phase):
    df = pd.read_csv(f'{eyetracking_dataset_path}/metadata_phase_{phase}.csv')
    return df['image'].unique().tolist()

def download_mimic_files_from_list(username, list_of_files, destination):
    with open('temp_download_mimic.txt', 'w') as f:
        for item in list_of_files:
            f.write("%s\n" % item)
    pathlib.Path(destination).mkdir(parents=True, exist_ok=True) 
    os.system(f'wget -r -c -np -nc --user {username} --ask-password -i ./temp_download_mimic.txt --directory-prefix={destination}');

parser = argparse.ArgumentParser()
parser.add_argument('--username', type=str, required=True,
                    help='PhysioNet username')
args = parser.parse_args()

# download_mimic_files_from_list(args.username, get_list_images_dicom(1) + get_list_images_dicom(2), dicom_path)

list_jpg = []
with open('image_all_paths.txt') as f:
    for line in f:
        list_jpg.append(line.rstrip())

# download_mimic_files_from_list(args.username, list_jpg, jpg_path)

tables = ['physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv.gz',
'physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv.gz']

download_mimic_files_from_list(args.username, tables, mimic_tables_dir)
for filename in tables:
    with gzip.open(f'{mimic_tables_dir}/{filename}', 'rb') as f_in:
        with open(f'{mimic_tables_dir}/{filename.split("/")[-1][:-3]}', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)