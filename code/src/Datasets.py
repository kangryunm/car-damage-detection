### lib import
import os
import glob
from torch.utils.data import Dataset
import torch
import cv2

import numpy as np
from pycocotools.coco import COCO
import albumentations as A

from src.Utils import customizedAnnToMask
import matplotlib.pyplot as plt
###


class Datasets(Dataset):
    def __init__(self, data_dir, mode, size, one_channel=False, transform=None, label=None, img_base_path=None):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.coco = COCO(data_dir)
        self.transform = transform
        self.one_channel = one_channel
        self.img_base_path = img_base_path if img_base_path else 'rst'
        
        print(self.coco.info())
        
        if mode in ('train','test'):
            self.img_ids = self.coco.getImgIds()
        else:
            self.img_ids = np.random.choice(self.coco.getImgIds(), 300, replace = False)
        if label is not None:  # damage: damage 종류 index
            self.label = label
        
        self.size = size
        if self.size:
            self.resize = A.Compose([A.Resize(width=self.size, height=self.size)])
        

    
    def __getitem__(self, index: int):
        image_id = int(self.img_ids[index])
        image_infos = self.coco.loadImgs(image_id)[0]
        # {'id': 1, 'width': 800, 'height': 600, 'file_name': '0001389_as-0067762.jpg'}

        # load image
        images = cv2.imread(os.path.join(self.img_base_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB) # w * h * c = (600, 800, 3)

        # load label
        if self.mode in ("train","val"):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])  # [1, 2, 3, 4, 5]
            anns = self.coco.loadAnns(ann_ids)
            

            masks = np.zeros((image_infos["height"], image_infos["width"]))
            
            ## damage
            if self.one_channel: 
                for ann in anns:
                    if ann['category_id'] == self.label:
                        masks = np.maximum(customizedAnnToMask(ann, image_infos), masks)
                        
                masks = masks.astype(np.float32)
                plt.imshow(masks, cmap='gray', vmin=0, vmax=255)
                
            
            ## part
            else: 
                for ann in anns: ## 내가 말한 단계. 여기서 annotation mask 생성
                    pixel_value = ann['category_id'] + 1  ## 14 + 1 = 15 (id=0은 background)
                    masks = np.maximum(self.coco.annToMask(ann) * pixel_value, masks)
                
                ## 아래는 원래 있던 주석
                # masks[0][masks.sum(axis=0) == 0] = 1
                # masks = masks.astype(np.float32) # n_cls * w * h

        # transform 
        if self.transform is not None: 
            transformed = self.transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]
            
        elif self.size:
            if self.one_channel:
                transformed = self.resize(image=images, mask=masks)
                masks = transformed["mask"]
            else:
                # masks = masks.transpose([1,2,0]) # w * h * n_cls
                transformed = self.resize(image=images, mask=masks)
                masks = transformed["mask"]
            images = transformed["image"]
        
        
        images = images/255.
        images = images.transpose([2,0,1]) 
        # if not(self.one_channel):
            # masks = masks.transpose([2,0,1]) # n_cls * w * h
        # images, masks = torch.tenor(images).float(), torch.tensor(masks).long()
        # images = torch.tensor(images)
        return images, masks, image_infos['file_name']



    def __len__(self) -> int:
        return len(self.img_ids)

