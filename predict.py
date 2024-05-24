import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import json
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import tensorflow_hub as hub
import os

from PIL import Image

import Functions as func

tf.TF_ENABLE_ONEDNN_OPTS=0
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('imagePath', type=str, help='Image Path')
parser.add_argument('model', type=str, help='Model Path')
parser.add_argument('--top_k',default=3,type=int, help='top_k classes')
parser.add_argument('--category_names',default=None, help='map.json')

args = parser.parse_args()

imagePath=args.imagePath 
model=args.model 
top_k=args.top_k
category_names=args.category_names

if type(category_names)!= type(None) :
    with open(f'{category_names}', 'r') as f:
        class_names = json.load(f)
else :
    class_names = None

model= tf.keras.models.load_model('model.h5', custom_objects={'KerasLayer':hub.KerasLayer})

func.plot_probas(imagePath ,model , top_k,class_names) 