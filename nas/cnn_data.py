from os.path import isfile, join

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from fedot.core.models.data import InputData
from fedot.core.repository.task_types import TaskTypesEnum, MachineLearningTasksEnum
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from nas.breeds import BreedsEnum

def from_json(file_path, task_type: TaskTypesEnum = MachineLearningTasksEnum.classification, train_size=0.75):
    df_train = pd.read_json(file_path)
    Xtrain = get_scaled_imgs(df_train)
    Ytrain = np.array(df_train['is_iceberg'])
    df_train.inc_angle = df_train.inc_angle.replace('na', 0)
    idx_tr = np.where(df_train.inc_angle > 0)
    Ytrain = Ytrain[idx_tr[0]]
    Xtrain = Xtrain[idx_tr[0], ...]
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain, Ytrain, random_state=1, train_size=0.75)
    Xtr_more = get_more_images(Xtrain)
    Ytr_more = np.concatenate((Ytrain, Ytrain, Ytrain))
    train_input_data = InputData(idx=np.arange(0, len(Xtr_more)), features=Xtr_more, target=np.array(Ytr_more),
                                 task_type=task_type)
    test_input_data = InputData(idx=np.arange(0, len(Xtest)), features=Xtest, target=np.array(Ytest),
                                task_type=task_type)
    return train_input_data, test_input_data

def encode_labels(breed_types=BreedsEnum.breed_types, is_one_hot=False):
    breed_df = pd.DataFrame(breed_types, columns=['Breed_Types'])
    labelencoder = LabelEncoder()
    breed_df['Breed_Types_Cat'] = labelencoder.fit_transform(breed_df['Breed_Types'])
    if is_one_hot:
        encoder_to_one_hot = LabelEncoder()
        encoder_to_one_hot.fit(breed_types)
        encoded_Y = encoder_to_one_hot.transform(breed_types)
        # convert integers to dummy variables (i.e. one hot encoded)
        ohe_hot_breeds = np_utils.to_categorical(encoded_Y)
        return ohe_hot_breeds

    breed_dict = {}
    for i in breed_df.values:
        breed_dict[i[0]]=i[1]

    return breed_dict


def from_images(file_path, task_type: TaskTypesEnum = MachineLearningTasksEnum.classification, train_size=0.75):
    possible_breeds = BreedsEnum.breed_types
    encoded_categories = encode_labels(possible_breeds, is_one_hot=False)
    size = 100
    files = [f for f in os.listdir(file_path) if isfile(join(file_path, f))]
    files.sort()
    Xtrain = []
    Ytrain = []
    # Y_one_hot = np.zeros(shape=[len(files), ohe_hot_categories.shape[0]])
    dftrain = pd.read_csv('dogs/labels.csv', sep=",")
    dogs_values = {}
    for x in dftrain.values:
        dogs_values[x[0]] = x[1]
    for i in range(len(files)):
        filename = file_path + files[i]
        img_rgb = cv2.imread(filename)
        img_rgb = cv2.resize(img_rgb, (size, size))
        label_ind = files[i][:-4]
        name_dog = dogs_values[label_ind]
        if name_dog in possible_breeds:
            Xtrain.append(img_rgb)
            encoded_vals = encoded_categories[name_dog]
            Ytrain.append(encoded_vals)
        else:
            continue
    Xtrain = np.array(Xtrain)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain, Ytrain, random_state=1, train_size=0.8)
    Xtrain = np.array(Xtrain)
    Xtest = np.array(Xtest)
    train_input_data = InputData(idx=np.arange(0, len(Xtrain)), features=Xtrain, target=np.array(Ytrain),
                                 task_type=task_type)

    test_input_data = InputData(idx=np.arange(0, len(Xtest)), features=Xtest, target=np.array(Ytest),
                                task_type=task_type)

    return train_input_data, test_input_data

def get_scaled_imgs(df):
    imgs = []

    for i, row in df.iterrows():
        # make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2  # plus since log(x*y) = log(x) + log(y)

        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)


def get_more_images(imgs):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images
