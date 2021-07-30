mimic_tables_dir = './mimic_tables/' # location of the metadata tables of the mimic-cxr dataset. Inside this folder:'mimic-cxr-2.0.0-chexpert.csv' and 'mimic-cxr-2.0.0-split.csv'
jpg_path = './dataset_mimic_jpg/' # location of the full mimic-cxr-jpg dataset. Inside this folder: "physionet.org" folder
dicom_path = './dataset_mimic/' # location of the dicom images of the mimic-cxr dataset that were included in the eyetracking dataset. Inside this folder: "physionet.org" folder
h5_path = '/scratch/ricbl/new_ambuj/' # folder where to save hdf5 files containing the full resized mimic dataset to speed up training
eyetracking_dataset_path = './dataset_et/' # location of the eye-tracking dataset. Inside this folder: metadata csv and all cases folders
segmentation_model_path = './segmentation_model/' #Inside this folder: trained_model.hdf5, from https://github.com/imlab-uiip/lung-segmentation-2d