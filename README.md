# etsaliencymaps
Steps to reproduce the results of the paper:
- get access to the MIMIC-CXR (https://physionet.org/content/mimic-cxr/2.0.0/) and MIMIC-CXR-JPG (https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset.
- download the datasets (jpg and dicom versions). To download only the images that are used in the experiments, you may run the provided script:
python src/download_mimic.py --username=<your physionet username>
- Download trained_model.hdf5 from https://github.com/imlab-uiip/lung-segmentation-2d and put in a folder named segmentation_model in the root of the git repository.
- Download the eye-tracking dataset from ... and put in a folder named dataset_et in the root of the git repository.
- Check that the folders listed in the config_paths.py files are correct for your setup.
- Run te following scripts:
python -m src.mimic_generate_df
python -m src.generate_heatmap_eyetracking
python -m src.get_segmentation_baseline 
python -m src.get_center_bias
python -m src.train
python -m src.train --model_name=ag_sononet
- Copy the sononet and ag_sononet folders from the folders ./runs/<folders of the runs you would like to test> to the folder models_to_test in the root of the git repository. Then run:
python -m src.generate_heatmap_model --folder_to_load=models_to_test --folder_to_save=heatmaps_model
- To get the results from Table 1, of comparison between heatmaps, run:
python -m src.compare_heatmaps --load_folder=heatmaps_model --save_folder=results
- The generated csv files list the scores for each of the test examples. There is one csv file for each of the 3 methods: Thresholded, Weighted, Uniform. In all 3 csv files, the results for the baselines inter-observer and convex segmentation are the same. The results for the attention maps are the same in the 3 csv files too.
- To get the AUC classification scores listed by the end of section 3.2, run:
python -m src.train -m sononet -ep 1 -ex test_auc_sononet -v test -l ./runs/<folder of the run you would like to test>/sononet/best_model.pt -bs 1
python -m src.train -m ag_sononet -ep 1 -ex test_auc_ag_sononet -v test -l ./runs/<folder of the run you would like to test>/ag_sononet/best_model.pt -bs 1
- The AUC values will be in the folders ./runs/test_auc_sononet_<timestamp>/sononet/logs.csv and ./runs/test_auc_ag_sononet_<timestamp>/ag_sononet/logs.csv