import pandas as pd
import os
import multiprocessing
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import matplotlib.pyplot as plt

meta=pd.read_csv('data/ISIC-2017_Training_Data_metadata.csv')
labels=pd.read_csv('data/ISIC-2017_Training_Part3_GroundTruth.csv')

# Fill na with mean for age and mode for sex
meta.sex=meta.sex.replace('unknown', None)
meta.age_approximate=meta.age_approximate.replace('unknown', None)

meta.sex=meta.sex.fillna(meta.sex.mode())
meta.age_approximate=meta.age_approximate.fillna(meta.age_approximate.mean())

meta.sex=meta.sex.replace(['female', 'male'], [0,1])
meta.sex=meta.sex.astype('float')
meta.age_approximate=meta.age_approximate.astype('float')
dataset=pd.concat((meta, labels[[1,2]]), axis=1)

def labeler(row):
    if row['melanoma'] == 0.0 and row['seborrheic_keratosis']==0.0:
        return 2
    elif row['melanoma'] == 1.0 and row['seborrheic_keratosis']==0.0:
        return 0
    else:
        return 1

labels['class'] = labels.apply(labeler, axis=1)
y=np.array(labels[[3]].values)
mlb = MultiLabelBinarizer()
y_binarized=mlb.fit_transform(y)
age_sex_labels=np.array(meta[[1,2]])

def images_preprocess(directory='data/ISIC-2017_Training_Data/', savedir='data/melanoma_preprocessed/'):

    Melanoma_cnt=0
    Seborreic_ceratosis_cnt=0
    Nevus_cnt=0

    if not os.path.exists(savedir):
        os.makedirs(savedir+'0_Melanoma/'),
        os.makedirs(savedir+'1_Seborreic_ceratosis/'),
        os.makedirs(savedir+'2_Nevus/')

    for img, im_class in labels[[0,3]].values:

        if im_class==0:
            for subdir, _, files in os.walk('data/ISIC-2017_Training_Data/'):
                subdir=subdir.replace('\\','/')
                subdir_split=subdir.split('/')
                if str('superpixels') not in files:
                    image_path=str(subdir+'/'+str(img)+'.jpg')
                    image=plt.imread(image_path)
                    plt.imsave((savedir+'0_Melanoma/'+str(img)+'.jpg'),image)

                    Melanoma_cnt+=1
                    if Melanoma_cnt%50==0:
                        print(' - '*20)
                        print('{} images are saved in {}'.format(Melanoma_cnt, (savedir+'0_Melanoma/')))


        elif im_class==1:
            for subdir, _, files in os.walk('data/ISIC-2017_Training_Data/'):
                subdir=subdir.replace('\\','/')
                subdir_split=subdir.split('/')
                if str('superpixels') not in files:
                    image_path=str(subdir+'/'+str(img)+'.jpg')
                    image=plt.imread(image_path)
                    plt.imsave((savedir+'1_Seborreic_ceratosis/'+str(img)+'.jpg'), image)

                    Seborreic_ceratosis_cnt+=1
                    if Seborreic_ceratosis_cnt%50==0:
                        print(' - '*20)
                        print('{} images are saved in {}'.format(Seborreic_ceratosis_cnt, (savedir+'1_Seborreic_ceratosis/')))
        else:
            for subdir, _, files in os.walk('data/ISIC-2017_Training_Data/'):
                subdir=subdir.replace('\\','/')
                subdir_split=subdir.split('/')
                if str('superpixels') not in files:
                    image_path=str(subdir+'/'+str(img)+'.jpg')
                    image=plt.imread(image_path)
                    plt.imsave((savedir+'2_Nevus/'+str(img)+'.jpg'), image)

                    Nevus_cnt+=1
                    if Nevus_cnt%50==0:
                        print(' - '*20)
                        print('{} images are saved in {}'.format(Nevus_cnt, (savedir+'2_Nevus/')))

    with open(savedir+"class_weight.txt", "w") as text_file:
        text_file.write("0_Melanoma: {}, 1_Seborreic_ceratosis:{}, 2_Nevus:{}".format(Melanoma_cnt, Seborreic_ceratosis_cnt, Nevus_cnt))

print('''
  /\/\   ___| | __ _ _ __   ___  _ __ ___   __ _  /\ \ \/\ \ \
 /    \ / _ \ |/ _` | '_ \ / _ \| '_ ` _ \ / _` |/  \/ /  \/ /
/ /\/\ \  __/ | (_| | | | | (_) | | | | | | (_| / /\  / /\  /
\/    \/\___|_|\__,_|_| |_|\___/|_| |_| |_|\__,_\_\ \/\_\ \/
    ''')
print('Images preprocess')
print('-'*50)

images_preprocess()
