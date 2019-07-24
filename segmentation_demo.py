import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random
import os

from utils.image import load_image
from utils.image import random_transform

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE

#image = load_image('./data/montgomery/images/MCUCXR_0001_0.png')
#image = random_transform(image)
#
#mask_left = load_image('./data/montgomery/masks/left/MCUCXR_0001_0.png')
#mask_right = load_image('./data/montgomery/masks/right/MCUCXR_0001_0.png')
#mask = tf.logical_or(tf.greater(mask_left,0.5), tf.greater(mask_right,0.5))
#mask = tf.cast(mask, tf.float32)
#mask = random_transform(mask)

#plt.imshow(tf.squeeze(image), cmap='gray')
#plt.imshow(tf.concat([tf.clip_by_value(image+mask*0.1, 0.0, 1.0), image, image], axis=2))

#def load_montgomery(path):
#    filename = os.path.basename(path)
#    image_dirname = os.path.dirname(path)
#    dirname = os.path.split(image_dirname)[0]    
#        
#    mask_left_path = os.path.join(*[dirname, 'masks', 'left', filename])
#    mask_right_path = os.path.join(*[dirname, 'masks', 'right', filename])
#    
#    image = load_image(path)
#    mask_left = load_image(mask_left_path)
#    mask_right = load_image(mask_right_path)
#    
#    mask = tf.logical_or(tf.greater(mask_left,0.5), tf.greater(mask_right,0.5))
#    mask = tf.cast(mask, tf.float32)
#    
#    return image, mask
def mask_merge(left, right):
    mask = tf.logical_or(tf.greater(left,0.5), tf.greater(right,0.5))
    mask = tf.cast(mask, tf.float32)
    return mask

montgomery_root = pathlib.Path('./data/montgomery')

# read image paths using glob and convert them into string format
montgomery_image_paths = [str(path) for path in list(montgomery_root.glob('images/*.png'))]
montgomery_mask_left_paths = [str(path) for path in list(montgomery_root.glob('masks/left/*.png'))]
montgomery_mask_right_paths = [str(path) for path in list(montgomery_root.glob('masks/right/*.png'))]

# shuffle them in a random order
temp = list(zip(montgomery_image_paths, montgomery_mask_left_paths, montgomery_mask_right_paths)) # zip the three paths to make sure they are shuffled together
random.shuffle(temp) # shuffle
montgomery_image_paths, montgomery_mask_left_paths, montgomery_mask_right_paths = zip(*temp) # unzip them
montgomery_image_paths = list(montgomery_image_paths)           # unzipping converts the lists to tuples for some reason...
montgomery_mask_left_paths = list(montgomery_mask_left_paths)   # we are explicitly converting them back to lists
montgomery_mask_right_paths = list(montgomery_mask_right_paths) #

# Create paths with tf.data.Dataset
montgomery_image_path_ds = tf.data.Dataset.from_tensor_slices(montgomery_image_paths)
montgomery_mask_left_path_ds = tf.data.Dataset.from_tensor_slices(montgomery_mask_left_paths)
montgomery_mask_right_path_ds = tf.data.Dataset.from_tensor_slices(montgomery_mask_right_paths)

# Image and mask datasets
montgomery_image_ds = montgomery_image_path_ds.map(load_image, num_parallel_calls=AUTOTUNE)
montgomery_mask_left_ds = montgomery_mask_left_path_ds.map(load_image, num_parallel_calls=AUTOTUNE)
montgomery_mask_right_ds = montgomery_mask_right_path_ds.map(load_image, num_parallel_calls=AUTOTUNE)

# Merge left and right masks
montgomery_mask_ds = tf.data.Dataset.zip((montgomery_mask_left_ds, montgomery_mask_right_ds))
montgomery_mask_ds = montgomery_mask_ds.map(mask_merge, num_parallel_calls=AUTOTUNE)

# (image, mask) pair
montgomery_ds = tf.data.Dataset.zip((montgomery_image_ds, montgomery_mask_ds))

for n, pair in enumerate(montgomery_ds.take(4)):
    image, mask = pair
    plt.subplot(2,2,n+1)
    plt.imshow(tf.concat([tf.clip_by_value(image+mask*0.1, 0.0, 1.0), image, image], axis=2))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()




















