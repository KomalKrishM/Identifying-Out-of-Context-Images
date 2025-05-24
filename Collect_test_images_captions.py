# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:22:46 2025

@author: komal
"""

import torch
import torchvision
from torch.utils.data import Dataset

import json
import os 


class NewsClipDataset(Dataset):
    
    def __init__(self, visual_news_root_dir, news_clip_root_dir, split, transform):
        self.visual_news_root_dir = visual_news_root_dir
        
        self.news_clip_root_dir = news_clip_root_dir
        self.transform = transform
        self.visual_news_data_dict = json.load(open(os.path.join(self.visual_news_root_dir+"data.json")))
       
        self.visual_news_data_mapping = {ann["id"]: ann for ann in self.visual_news_data_dict}
        
        self.news_clip_data_dict = json.load(open(os.path.join(self.news_clip_root_dir,split+".json")))["annotations"]    
        
        
    def __len__(self):
        return len(self.news_clip_data_dict)   


    def __getitem__(self, idx):      
        if torch.is_tensor(idx):
            idx = idx.tolist()    
        
        label = 1 if self.news_clip_data_dict[idx]['falsified'] else 0

        visual_news_caption_item = self.visual_news_data_mapping[self.news_clip_data_dict[idx]["id"]]
        caption = visual_news_caption_item['caption']
        
        visual_news_image_item = self.visual_news_data_mapping[self.news_clip_data_dict[idx]["image_id"]]
        image_path = os.path.join(self.visual_news_root_dir, visual_news_image_item['image_path'])

        return label, image_path, caption
        
        
# collecting_test_data 
import shutil
from torch.utils.data import DataLoader

visual_news_root_dir = '/Users/komalkrishnamogilipalepu/Downloads/origin/' #"../visual_news/origin/"
news_clip_root_dir = '/Users/komalkrishnamogilipalepu/Downloads/OoC-multi-modal-fc-main/news_clippings/data/merged_balanced/' #"../news_clippings/data/merged_balanced/"
split = 'test'

test_dataset = NewsClipDataset(visual_news_root_dir, news_clip_root_dir, split, None)

def custom_collate_mismatch(batch): 
    labels = [item[0] for item in batch]
    imgs = [item[1] for item in batch] 
    captions_batch = [item[2] for item in batch] 
    return labels, imgs, captions_batch

test_labels = 'D:/OoC-multi-modal-fc-main/finetuning_clip/test_labels/test_labels.txt'
test_captions = 'D:/OoC-multi-modal-fc-main/finetuning_clip/test_captions/test_captions.txt'
test_images_path = 'D:/OoC-multi-modal-fc-main/finetuning_clip/test_images/'  
  
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn = custom_collate_mismatch, num_workers=0)
labels, images, captions_batch  = next(iter(test_dataloader))

with open(test_captions, 'w') as caption_file:    
    for i in range(len(images)):
        shutil.copy(images[i], test_images_path+images[i][-12:-8]+'_'+images[i][-7:-4] + '.jpg')
        caption_file.write(str(labels[i]) + ' ' + images[i][-12:-4] + ' ' + captions_batch[i] + '.' +'\n')
    

