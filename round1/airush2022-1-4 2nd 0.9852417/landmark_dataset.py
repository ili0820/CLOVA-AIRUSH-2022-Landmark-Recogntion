from torch.utils.data import Dataset
import csv
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd
from args import args
"""
label_file 구조
@ image_idx, class_idx, class_id, image_file_name
>> image_idx        : 이미지 파일별 고유 인덱스 번호 (0~이미지수-1)
>> class_idx        : 랜드마크별 고유 인덱스 번호 (0~랜드마크수-1)
>> class_id         : 랜드마크별 고유 id값
>> image_file_name  : 이미지 파일명
"""

class LandmarkRecognitionDataset(Dataset):
    def __init__(self, data_dir, label_file_path, transform=None, num_classes=619):
        if not label_file_path == None:
            csvfile = open(label_file_path, newline='', encoding='utf-8')
            csvread = csv.reader(csvfile, delimiter=',')
            self.labels = list(csvread)
            csvfile.close()
        else:
            self.labels = None
        self.data_list = []
        for (dirpath, dirnames, filenames) in os.walk(data_dir):
            self.data_list.extend(filenames)
            break

        self.data_list.sort()
        self.data_dir = data_dir
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        if self.labels == None:
            return len(self.data_list)
        else:
            return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
                idx = idx.tolist()

        if self.labels == None:
            image_path = os.path.join(self.data_dir, self.data_list[idx])
            image = Image.open(image_path).convert('RGB')

            if not self.transform == None:
                image = self.transform(image)
            else:
                transform = transforms.Compose([transforms.ToTensor()])
                image = transform(image)

            image_id = str(int(self.data_list[idx].split("_")[0]))

            return (image, torch.tensor([]), image_id, torch.tensor([]))
        else:
            image_path = os.path.join(self.data_dir, self.labels[idx][4])
            image = Image.open(image_path).convert('RGB')

            if not self.transform == None:
                image = self.transform(image)
            else:
                transform = transforms.Compose([transforms.ToTensor()])
                image = transform(image)

            image_id = self.labels[idx][0]
            category_id = int(self.labels[idx][1])
            
            label = np.zeros((self.num_classes))
            label[category_id] = 1

            return (image, torch.tensor(category_id), image_id, torch.tensor(label))
    def getmargin(self):

        tmp = np.sqrt(1 / np.sqrt(pd.DataFrame(self.labels)[2].value_counts().sort_index().values))
        margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05
        return margins
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader 
import albumentations as A
class LandmarkRecognitionDataset2(Dataset):
    def __init__(self, csv, aug=None, normalization='simple', is_test=False): 
        self.labels = csv.class_id.values
        self.csv = csv.filepath.values
        self.aug = aug
        self.normalization = normalization
        self.is_test = is_test

    def __getitem__(self, index):
        img_path = self.csv[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.aug:
            img = self.augment(img)
        img = img.astype(np.float32)

        if self.normalization:
            img = self.normalize_img(img)

        tensor = self.to_torch_tensor(img)
        if self.is_test:
            feature_dict = {'idx':torch.tensor(index),
                            'input':tensor}
        else:
            # 'index','image_idx','landmark_id','class_id','id'
            # target = torch.tensor(self.labels[index])
            category_id = int(self.labels[index])
            label = np.zeros((args.n_classes))
            label[category_id] = 1

            feature_dict = {'idx':torch.tensor(index),
                            'input':tensor,
                            'target':torch.tensor(category_id)}
        return feature_dict

    def __len__(self): 
        return len(self.csv)

    def augment(self,img):
        img_aug = self.aug(image=img)['image']
        return img_aug.astype(np.float32)

    def normalize_img(self,img):
        if self.normalization == 'imagenet':
            mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
            std = np.array([58.395   , 57.120, 57.375   ], dtype=np.float32)
            img = img.astype(np.float32)
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
        elif self.normalization == 'inception':
            mean = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            img = img.astype(np.float32)
            img = img/255.
            img = img-mean
            img = img*np.reciprocal(std, dtype=np.float32)
        else:
            pass
        return img
    
    def to_torch_tensor(self,img):
        return torch.from_numpy(img.transpose((2, 0, 1)))

class LandmarkRecognitionDataset3(Dataset):
    def __init__(self, data_dir, aug=None, normalization='simple', is_test=False):
        self.data_dir=data_dir
        self.labels = None
        self.data_list = []
        for (dirpath, dirnames, filenames) in os.walk(data_dir):
            self.data_list.extend(filenames)
            break
        
        self.aug = aug
        self.normalization = normalization
        self.is_test = is_test
        self.data_list.sort()



    def __getitem__(self, index):
        # img_path = self.csv[index]
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_path = os.path.join(self.data_dir, self.data_list[index])
            
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_id = str(int(self.data_list[index].split("_")[0]))
        if self.aug:
            img = self.augment(img)
        img = img.astype(np.float32)

        if self.normalization:
            img = self.normalize_img(img)

        tensor = self.to_torch_tensor(img)

        feature_dict = {'idx':torch.tensor(index),
                            'input':tensor,
                            'image_id':image_id}

        return feature_dict

    def __len__(self): 
        return len(self.data_list)

    def augment(self,img):
        img_aug = self.aug(image=img)['image']
        return img_aug.astype(np.float32)

    def normalize_img(self,img):
        if self.normalization == 'imagenet':
            mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
            std = np.array([58.395   , 57.120, 57.375   ], dtype=np.float32)
            img = img.astype(np.float32)
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
        elif self.normalization == 'inception':
            mean = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            img = img.astype(np.float32)
            img = img/255.
            img = img-mean
            img = img*np.reciprocal(std, dtype=np.float32)
        else:
            pass
        return img
    
    def to_torch_tensor(self,img):
        return torch.from_numpy(img.transpose((2, 0, 1)))