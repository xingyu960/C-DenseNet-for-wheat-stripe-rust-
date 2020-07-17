# coding = utf-8
import sys

# # Call the subroutine under the folder
sys.path.append('/C-DenseNet/dataset')
sys.path.append('/C-DenseNet/utils')
sys.path.append('/C-DenseNet/models')

import os
import torch
import torch.utils.data as data
import PIL.Image as Image
from data_aug import *
import cv2
import numpy as np
import random
import glob
import pandas as pd

#The alphabetic name of the tag
defect_label_order = ['zero', 'one', 'two', 'three', 'four', 'five']
# Corresponds to the alphabetic name of tags
defect_code = {
    'zero':  'zero',
    'one':  'one',
    'two':  'two',
    'three':  'three',
    'four': 'four',
    'five': 'five'
}
# Corresponds to number tags
defect_label = {
    'zero':  '0',
    'one':  '1',
    'two':    '2',
    'three':  '3',
    'four':  '4',
    'five': '5'
}


label2defect_map = dict(zip(defect_label.values(), defect_label.keys()))
# Get the image root
def get_image_pd(img_root):
    # Use the glob command to get the picture list (/ * number is determined according to the file composition)
    img_list = glob.glob(img_root + "/*/*.jpg")
    # The dictionary of picture list is constructed by dataframe instruction, that is, the serial number of picture list corresponds to its path one by one
    image_pd = pd.DataFrame(img_list, columns=["ImageName"])
    # Gets the folder name, which can also be considered the label name
    image_pd["label_name"]=image_pd["ImageName"].apply(lambda x:x.split("/")[-2])
    # Converts the label name to a numeric tag
    image_pd["label"]=image_pd["label_name"].apply(lambda x:defect_label[x])
    print(image_pd["label"].value_counts())
    return image_pd

# Image set
class dataset(data.Dataset):
    def __init__(self, anno_pd, transforms=None,debug=False,test=False):
        self.paths = anno_pd['ImageName'].tolist()
        self.labels = anno_pd['label'].tolist()
        self.transforms = transforms
        self.debug=debug
        self.test=test
    # Number of images returned
    def __len__(self):
        return len(self.paths)
    # Get each picture
    def __getitem__(self, item):
        img_path =self.paths[item]
        img_id =img_path.split("/")[-1]
        img =cv2.imread(img_path) #BGR
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   # [h,w,3]  RGB
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.labels[item]
        # label[label > 5] = ignore_label
        # label[label < 0] = ignore_label
        if self.test:
            return torch.from_numpy(img).float(), int(label)
        else:
            return torch.from_numpy(img).float(), int(label)

# Organize pictures
def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label


