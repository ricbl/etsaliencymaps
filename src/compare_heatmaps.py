#!/usr/bin/env python
# coding: utf-8

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-sf','--save_folder', type=str, default=None)
ap.add_argument('-lf','--load_folder', type=str, default=None)
args = ap.parse_args()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from PIL import Image
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

model_heatmaps_folder = args.load_folder
output_folder = args.save_folder
save_images = False

import pathlib
pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True) 

def get_cases(phase, et_path):
    this_trial_this_index_image = et_path.split('/')[-1]
    this_trial = int(this_trial_this_index_image.split('_')[0])
    index_image = this_trial_this_index_image.split('_')[1].split('.')[0]
    to_return_intrauser = {'trial':this_trial, 'index_image':index_image, 'et_path':et_path, 'phase':phase}
    return to_return_intrauser

def smooth_auc(im1, im2, average_map):
    im1 = im1/np.sum(im1)
    a = np.ones(im1.shape)
    return smooth_shuffled_auc(im1, im2, a/np.sum(a))

def smooth_shuffled_auc(im1, im2, average_map):
    aucs = []
    im1 = im1/np.sum(im1)
    for i in range(1):
        total_samples = 1000
        input1 = np.zeros(total_samples*2).astype(np.float32)
        input2 = np.zeros(total_samples*2).astype(np.float32)
        input1[total_samples:]=1
        input2[input1==0]= np.random.choice(
                          im2.flatten(), 
                          total_samples,
                          p=average_map.flatten()
                        )
        input2[input1==1]= np.random.choice(
                          im2.flatten(), 
                          total_samples,
                          p=(im1).flatten()
                        )
        auc = roc_auc_score(input1.flatten(), input2.flatten())
        aucs.append(auc)
    return auc

def shuffled_normalised_corr(im1, im2, average_map):
    average_map = average_map.astype(np.float32)
    return normalised_corr(im1, im2, None) - normalised_corr(average_map, im2, None)

def normalised_corr(a,v, average_map): 
    return np.mean((a-np.mean(a))*(v-np.mean(v)))/np.std(v)/np.std(a)

def get_average_et_scores(i, case, averageuser_df, mixing_method, metrics):
    results_averageuser_df =[]
    results_averageintrauser_df = []
    trial = case[0]
    this_trial = trial
    phase = case[1]
    print(i, trial, phase)
    loaded_pickle_base = np.load(f'./heatmaps_subtraction_{phase}_1/{this_trial}.npy', allow_pickle=True).item()
    et_base = loaded_pickle_base['np_image']
    
    same_trial_all_users = averageuser_df[(averageuser_df['trial']==trial) & (averageuser_df['phase']==phase)]
    if len(same_trial_all_users)>=4:
        assert(len(same_trial_all_users)<=5)
        et_heatmaps = {}
        previous_img_path = None
        for _,row in same_trial_all_users.iterrows():
            loaded_pickle = np.load(row['et_path'], allow_pickle=True).item()
            if previous_img_path is not None:
                print(previous_img_path)
                print(loaded_pickle['img_path'])
                assert(loaded_pickle['img_path']==previous_img_path)
            previous_img_path = loaded_pickle['img_path']
            assert(this_trial==loaded_pickle['trial'])
            et_heatmaps[row['index_image']] = loaded_pickle['np_image']

        average_et = np.average(np.stack(et_heatmaps.values(), axis = 0), axis = 0)
        heatmaps_path = [f'./{model_heatmaps_folder}/{mixing_method}/sononet/phase_{phase}_0.98/{this_trial}.npy',
        f'./{model_heatmaps_folder}/{mixing_method}/ag_sononet/phase_{phase}_0.98/{this_trial}.npy',
        f'./{model_heatmaps_folder}/{mixing_method}/ag_sononet/phase_{phase}_0.98/{this_trial}_am1.npy',
        f'./{model_heatmaps_folder}/{mixing_method}/ag_sononet/phase_{phase}_0.98/{this_trial}_am2.npy',
        ]
        
        previous_image_path = None
        np_images = []
        for heatmap_path in heatmaps_path:
            loaded_pickle = np.load(heatmap_path, allow_pickle=True).item()
            img_path = loaded_pickle['img_path']
            assert(this_trial==loaded_pickle['trial'])
            print(previous_img_path)
            print(img_path)
            assert(previous_img_path == img_path)
            previous_image_path = img_path
            np_images.append(loaded_pickle['np_image'])
        img_path = previous_image_path.split('/')[-1].split('.')[-2]
        img_path = f'segmentations_convex/segmentation_{img_path}.png'
        img = Image.open( img_path )
        data = np.array( img, dtype=np.float32 )[:,:,0]
        np_images.append(data)
        if len(same_trial_all_users)==5:
            for index_image in et_heatmaps.keys():
                this_user_et  = et_heatmaps[index_image]
                other_user_ets  = [et_heatmaps[key] for key in et_heatmaps.keys() if key != index_image]
                assert(len(other_user_ets)==4)
                average_other_users = np.average(np.stack(other_user_ets, axis = 0), axis = 0)
                
                if save_images:
                    for index_map, map in {'user':this_user_et, 'other_users':average_other_users, 'convex':np_images[4], 'center_bias':et_base, 'sonocam':np_images[0], 'agcam':np_images[1], 'am1':np_images[2], 'am2':np_images[3]}.items():
                        plt.imshow(plt.imread(previous_image_path), cmap='gray')
                        plt.imshow(map, cmap='jet', alpha = 0.3)
                        plt.axis('off')
                        plt.savefig(f'{output_folder}/trial_{this_trial}_phase_{phase}_{index_image}_{mixing_method}_map_{index_map}.png', bbox_inches='tight', pad_inches = 0)
                                            
                for metric, metric_fn in metrics.items():
                    user_user_score = metric_fn(average_other_users, this_user_et, et_base)
                    score_sono_4= metric_fn(average_other_users, np_images[0], et_base)
                    score_ag_4= metric_fn(average_other_users, np_images[1], et_base)
                    score_am1_4 = metric_fn(average_other_users, np_images[2], et_base)
                    score_am2_4 = metric_fn(average_other_users, np_images[3], et_base)
                    score_convex_4 = metric_fn(average_other_users, np_images[4],et_base)
                    results_averageintrauser_df.append({'trial':trial,'phase':phase,'index_image':index_image, 'metric':metric,'score_interobserver':user_user_score,'score_sono':score_sono_4, 'score_ag':score_ag_4 , 'score_am1':score_am1_4, 'score_am2':score_am2_4, 'score_convex':score_convex_4})
    return results_averageintrauser_df

def main():
    metrics = { 'sncc':shuffled_normalised_corr, 'ncc':normalised_corr, 'auc':smooth_auc, 'sauc':smooth_shuffled_auc, }
    
    mixing_methods = ['uniform','weighted_classes', 'thresholded']
    # ### Heatmaps from radiologists' eyetracking data
    def sorter(item):
        return f'{int(item.split("/")[-1].split("_")[0]):04}{item.split("/")[-1].split("_")[1]}'
    list_et = {1: sorted(glob.glob( './heatmaps_phase_1_1/*npy'),key=sorter), 2: sorted(glob.glob('./heatmaps_phase_2_1/*npy'),key=sorter)}

    for mixing_method in mixing_methods:
        # ## Comparing Sononet with Radiologists    
        list_intrauser_df = []
        for phase in [1,2]:
            list_intrauser_df_ = Parallel(n_jobs=32)(delayed(get_cases)(phase, et_path) for i,et_path in enumerate(list_et[phase]))
            list_intrauser_df += list_intrauser_df_
        
        averageuser_df = pd.DataFrame(list_intrauser_df)
        averageuser_df['case'] = list(zip(averageuser_df.trial, averageuser_df.phase))
        cases = pd.unique(averageuser_df['case'])
        list_average_intrauser = Parallel(n_jobs=1)(delayed(get_average_et_scores)(i,case, averageuser_df, mixing_method,metrics) for i,case in enumerate(cases))
        list_average_intrauser = [item for sublist in list_average_intrauser for item in sublist]
        pd.DataFrame(list_average_intrauser).to_csv(f'{output_folder}/results_{mixing_method}_df.csv', index=False)

if __name__ == "__main__":
   main()
