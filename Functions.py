# TODO: Make all necessary imports.
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import tensorflow_hub as hub
import os


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import tensorflow_hub as hub
import os

from PIL import Image

def process_image(image ):
    image_size=224
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image=image/ 255
    image=np.expand_dims(image, axis=0)
    return image


def predict(image_path, model, top_k) :
    im = Image.open(image_path)
    im = np.asarray(im)
    processedImage=process_image(im)
    print(processedImage.shape)
    prediction = model.predict(processedImage)[0]
    classes = np.argsort(prediction)[-top_k:]
    probs=np.sort(prediction)[-top_k:]
    return probs, classes,processedImage

def plot_probas(imagePath ,model , top_k,class_names=None) : 
    probs, classes ,processedImage = predict(imagePath, model, top_k)
    ticks=len(probs)
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(processedImage.squeeze() )
    ax1.axis('off')
    ax2.barh(np.arange(ticks), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(ticks))
    if type(class_names)!= type(None) : 
        classNames=[class_names[str(x)] for x in classes]
        ax2.set_yticklabels(classNames, size='small')
    else : 
        ax2.set_yticklabels([str(x) for x in classes], size='small')

    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()