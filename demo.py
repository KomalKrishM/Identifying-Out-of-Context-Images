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
import clip_classifier
import clip

from PIL import Image

parser = argparse.ArgumentParser(description='Training using the precomputed embeddings')
##### locations #####
parser.add_argument('--exp_folder', type=str, default='/Users/komalkrishnamogilipalepu/Downloads/OoC-multi-modal-fc-main/exp',
                    help='path to the folder to log the output and save the models')

###### model details ########                    
parser.add_argument('--pdrop', type=float, default=0.5,
                    help='dropout probability')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)

#### settings of the model ####
device = "mps" if torch.backends.mps.is_built() else "cpu"
model_settings = {'pdrop': args.pdrop}
base_clip, preprocess = clip.load("ViT-B/32", device=device)
classifier_clip = clip_classifier.ClipClassifier(model_settings,base_clip)
classifier_clip.to(device)


def process_img_caption(image_path, caption, transform):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        pil_img = img.convert('RGB')

    transform_img = transform(pil_img)
    caption_tokenized = clip.tokenize(caption)

    return transform_img, caption_tokenized


def test(image, caption, device):

    img, cap = process_img_caption(image, caption, preprocess)
    img = img.unsqueeze(0)
    img = img.to(device)
    cap = cap.to(device)

    output = classifier_clip(img, cap)
    # print(output)
    # print(torch.sigmoid(output))
    pred = torch.sigmoid(output) >= 0.5

    return pred


test_image_path_1 = '/Users/komalkrishnamogilipalepu/Downloads/OoC-multi-modal-fc-main/finetuning_clip/test_images/0223_867.jpg'
test_caption_1 = 'The iconic Taj Mahal is a Unesco World Heritage site and was built in the 17th Century.'
test_image_path_2 = '/Users/komalkrishnamogilipalepu/Downloads/OoC-multi-modal-fc-main/finetuning_clip/test_images/0670_067.jpg'
test_caption_2 = 'President Obama arrives in Ohio.'
test_image_path_3 = '/Users/komalkrishnamogilipalepu/Downloads/OoC-multi-modal-fc-main/finetuning_clip/test_images/0680_114.jpg'
test_caption_3 = 'President Obama delivers the State of the Union to a joint session of Congress and honored guests.'
test_image_path_4 = '/Users/komalkrishnamogilipalepu/Downloads/OoC-multi-modal-fc-main/finetuning_clip/test_images/0204_397.jpg'
test_caption_4 = 'Members of the media watch the Democratic presidential debate at Drake University.'
test_image_path_5 = '/Users/komalkrishnamogilipalepu/Downloads/OoC-multi-modal-fc-main/finetuning_clip/test_images/0050_618.jpg'
test_caption_5 = 'Romney speaks at a Salt Lake City fundraiser on Sept 18.'
test_image_path_6 = '/Users/komalkrishnamogilipalepu/Downloads/OoC-multi-modal-fc-main/finetuning_clip/test_images/0705_423.jpg'
test_caption_6 = 'Hillary Clinton speaks at a book signing for Hard Choices at a Barnes Noble in New York City on June 10 2014 the day of the book s release.'
falsified_true_labels = [True, True, False, False, True, False] #[falsified_1, falsified_2, falsified_3, falsified_4, falsified_5, falsified_6]

img_paths = [test_image_path_1, test_image_path_2, test_image_path_3, test_image_path_4, test_image_path_5, test_image_path_6]
captions = [test_caption_1, test_caption_2, test_caption_3, test_caption_4, test_caption_5, test_caption_6]
falsified_pred = []
# Load the best saved model if exists and run for the test data.
if os.path.isfile(os.path.join(args.exp_folder, 'best_model_acc.pth.tar')):
    checkpoint = torch.load(os.path.join(args.exp_folder, 'best_model_acc.pth.tar'), map_location=device)
    classifier_clip.load_state_dict(checkpoint['state_dict'])
    clip.model.convert_weights(classifier_clip)
    print("=> loaded checkpoint: '{}')".format(os.path.join(args.exp_folder, 'best_model_acc.pth.tar')))

    # Run on test data.
    for img_path, caption in zip(img_paths, captions):
        pred = test(img_path, caption, device)
        falsified_pred.append(pred.item())
        # print("falsified prediction:", pred)

    print("Falsified predicted labels:", falsified_pred)
    print("Falsified true labels:", falsified_true_labels)