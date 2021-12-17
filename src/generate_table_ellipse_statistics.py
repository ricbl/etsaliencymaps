import pandas as pd
from config_paths import eyetracking_dataset_path
from collections import defaultdict
import shapely
import shapely.geometry as geometry

def create_ellipse(coords_box_):
    coords_box = []
    coords_box.append((coords_box_[0]+coords_box_[2])/2)
    coords_box.append((coords_box_[1]+coords_box_[3])/2)
    coords_box.append(abs(coords_box_[0]-coords_box_[2])/2)
    coords_box.append(abs(coords_box_[1]-coords_box_[3])/2)
    rect = coords_box
    center = (rect[0],rect[1])
    axis = (rect[2],rect[3])
    point = geometry.point.Point(center).buffer(1)
    ellipse = shapely.affinity.scale(point, int(axis[0]), int(axis[1]))
    return ellipse

labels = {'Emphysema':'','High lung volume / emphysema':'',
          'Fibrosis':'parenchymal','Interstitial lung disease':'parenchymal',
          'Enlarged cardiac silhouette':'cardiomediastinal',
          'Pneumothorax':'pleural',
          'Atelectasis':'parenchymal',
          'Consolidation':'parenchymal',
          'Fracture':'','Acute fracture':'',
          'Groundglass opacity':'parenchymal',
          'Edema':'parenchymal','Pulmonary edema':'parenchymal',
          'Abnormal mediastinal contour':'cardiomediastinal',
          'Wide mediastinum':'cardiomediastinal',
          'Airway wall thickening':'parenchymal',
          'Mass':'parenchymal', 'Nodule':'parenchymal', 
          'Pleural effusion':'pleural','Pleural thickening':'pleural',
          'Support devices':'',
          'Enlarged hilum':'cardiomediastinal',
          'Hiatal hernia':'',
          'Lung nodule or mass':'parenchymal',
          'Pleural abnormality':'pleural',
          'Other':''}

grouped_labels_area = defaultdict(list)
grouped_labels_certainty = defaultdict(list)
labels_area = defaultdict(list)
labels_certainty = defaultdict(list)
abn_area = []
abn_certainty = []
results_csv = pd.DataFrame()
certainty_to_entropy = {1:0.46899559, 2:0.81127812, 3:1., 4:0.81127812, 5:0.46899559}
# for each phase
for phase in [1,2]:
    # load metadata from eyetracking dataset
    df = pd.read_csv(f'{eyetracking_dataset_path}/metadata_phase_{phase}.csv')
    # filter images according to rule of all eyetracking data must be present
    df = df[df['eye_tracking_data_discarded']==False]
    all_images = df['image'].unique()
    for image in all_images:
        same_trial_all_users = df[df['image']==image]
        if len(same_trial_all_users)>=4:
            assert(len(same_trial_all_users)<=5)
            # For each image
            for _, reading in same_trial_all_users.iterrows():
                # load box metadata
                df_box = pd.read_csv(f'{eyetracking_dataset_path}/{reading["id"]}/anomaly_location_ellipses.csv')
                
                # for each box
                for _, box in df_box.iterrows():
                    # calculate area of the ellipsis
                    ellipse = create_ellipse(box[['xmin','ymin','xmax','ymax']])
                    area = ellipse.area
                    certainty = certainty_to_entropy[box['certainty']]
                    # for each label of the box
                    added_abn = False
                    added_group = defaultdict(lambda: False)
                    for label in labels:
                        if label in box.keys():
                            if box[label]:
                                # add area to the list of areas for the dictionary item of a label
                                if not added_abn:
                                    abn_area.append(area)
                                    abn_certainty.append(certainty)
                                    added_abn = True
                                    results_csv = results_csv.append({'label':'abnormality', 'area': area, 'certainty': certainty}, ignore_index=True)
                                    
                                if labels[label]!='':
                                    if not added_group[labels[label]]:
                                        grouped_labels_area[labels[label]].append(area)
                                        grouped_labels_certainty[labels[label]].append(certainty)
                                        added_group[labels[label]] = True
                                        results_csv = results_csv.append({'label':labels[label], 'area': area, 'certainty': certainty}, ignore_index=True)
                                labels_area[label].append(area)
                                labels_certainty[label].append(certainty)
                                results_csv = results_csv.append({'label':label, 'area': area, 'certainty': certainty}, ignore_index=True)
                    
#For each label
for label in labels:
    print(label)
    print(len(labels_area[label]))
    if (len(labels_area[label])>0):
        print(sum(labels_area[label])/len(labels_area[label]))
        print(sum(labels_certainty[label])/len(labels_certainty[label]))
    #calculate certainty average
    # calculate the average area

#group labels according to subdivisions: abnormal, parenchymal, ...
for label in grouped_labels_area:
    print(label)
    print(len(grouped_labels_area[label]))
    print(sum(grouped_labels_area[label])/len(grouped_labels_area[label]))
    print(sum(grouped_labels_certainty[label])/len(grouped_labels_certainty[label]))
    # calculate averages with this new grouping

print('Abnormal')
print(len(abn_area))
print(sum(abn_area)/len(abn_area))
print(sum(abn_certainty)/len(abn_certainty))

results_csv.to_csv('./ellipses_statistics_by_label.csv')