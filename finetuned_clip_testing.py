# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 22:57:06 2025

@author: user
"""

import json 
import os 
import torch
import torchvision
import torch.nn as nn
import numpy as np 

import argparse
import io
import clip_classifier
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import time 
import clip 

from PIL import Image

parser = argparse.ArgumentParser(description='Training using the precomputed embeddings')
##### locations #####  
parser.add_argument('--visual_news_root', type=str, default='D:/origin/',
                    help='location to the root folder of the visualnews dataset')
# parser.add_argument('--visual_news_root_2', type=str, default='C:/Users/user/Downloads/origin/',
#                     help='location to the root folder of the visualnews dataset')
parser.add_argument('--news_clip_root', type=str, default='D:/OoC-multi-modal-fc-main/news_clippings/data/merged_balanced/',
                    help='location to the root folder of the clip dataset')               
parser.add_argument('--exp_folder', type=str, default='D:/OoC-multi-modal-fc-main/exp',
                    help='path to the folder to log the output and save the models')
                    
###### model details ########                    
parser.add_argument('--pdrop', type=float, default=0.5,
                    help='dropout probability')


##### Training details #####
parser.add_argument('--batch_size', type=int, default=64,
                    help='dimension of domains embeddings') 
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of data loaders workers') 
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--log_interval', type=int, default=200,
                    help='how many batches')
parser.add_argument('--resume', type=str, default = '', help='path to model')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='which optimizer to use')
parser.add_argument('--lr_clip', type=float, default= 5e-7,
                    help='learning rate of the clip model')
parser.add_argument('--lr_classifier', type=float, default=5e-5,
                    help='learning rate of the clip model')                    
parser.add_argument('--sgd_momentum', type=float, default=0.9,
                    help='momentum when using sgd')                      
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--wdecay', default=1.2e-6, type=float,
                        help='weight decay pow (default: -5)')
                    
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#### settings of the model ####
model_settings = {'pdrop': args.pdrop}
base_clip, preprocess = clip.load("ViT-B/32", device="cuda")
classifier_clip = clip_classifier.ClipClassifier(model_settings,base_clip)
classifier_clip.cuda()


#resume training
stored_loss = 100000000   
stored_acc = 0

classifier_list = ['classifier.weight', 'classifier.bias']
classifier_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in classifier_list, classifier_clip.named_parameters()))))
base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in classifier_list, classifier_clip.named_parameters()))))

 
#define loss function
criterion = nn.BCEWithLogitsLoss()

params = list(classifier_clip.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


def process_img_caption(image_path, caption, transform):
    
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        pil_img = img.convert('RGB')
        
    transform_img = transform(pil_img)
    caption_tokenized = clip.tokenize(caption)
    
    return transform_img, caption_tokenized

def test(image, caption):
    
    # classifier_clip.eval()
    img, cap = process_img_caption(image, caption, preprocess) 
    img = img.reshape([1, 3, 224, 224])
    img = img.cuda()
    cap = cap.cuda()
    # print(img.shape)
    # print(cap.shape)
    
    output = classifier_clip(img, cap)
    # loss = criterion(output, label)    
    print(output)
    pred = torch.sigmoid(output) >= 0.5
    
    return pred

# image_path = "D:/Previous Downloads/Soma mam lab/flickr30k-images/1624481.jpg" #" 2656351.jpg
image_path = "D:/OoC-multi-modal-fc-main/GenAI/US Underwater Transportation project.jpg" #Kid_Infant_food.jpg" #cat_swimming.jpg"     # raffle shows on independence day.jpg"              # Trump_Sonia.jpg"
# caption = 'A young girl in red Snoopy pants is holding a very small baby on her lap'
# caption = 'A young girl scarying the infant '
# caption = ' The army people forgot their duty to save the country '
# 'Many army men in a line holding guns in green suits walking down a concrete road with a big building in the background'  
# caption = "Donald Trump having dinner with Sonia Gandhi"
# caption = "rabbit is swimming with dolphins in water"
# caption = "rabbit is playing with ducks in water"
# caption = "cat is swimming in the water"
# caption = "Kid is eating food with infant"
caption = "United states started underwater transportation project"
# caption = "group of children are celebrating united states on independence day"

# Load the best saved model if exists and run for the test data.
if os.path.isfile(os.path.join(args.exp_folder, 'best_model_acc.pth.tar')):
    checkpoint = torch.load(os.path.join(args.exp_folder, 'best_model_acc.pth.tar'))
    classifier_clip.load_state_dict(checkpoint['state_dict'])
    clip.model.convert_weights(classifier_clip)
    print("=> loaded checkpoint: '{}')".format(os.path.join(args.exp_folder, 'best_model_acc.pth.tar')))
    # Run on test data.
    prediction = test(image_path, caption)
    print('=' * 89)
    print('Prediction:', prediction)
    print('=' * 89)