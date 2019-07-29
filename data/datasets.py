'''

'''
import tensorflow as tf
import os
import shutil
import pathlib
import random
import numpy as np
from utils.image import load_image       # maybe utils is not an appropriate location
from utils.image import affine_transform
from utils.image import lens_distortion

AUTOTUNE = tf.data.experimental.AUTOTUNE

def _mask_merge(left, right):
    mask = tf.logical_or(tf.greater(left,0.5), tf.greater(right,0.5))
    mask = tf.cast(mask, tf.float32)
    return mask

# TODO(stephen-baek): Can user set the distortion parameters?
def random_transform(image, mask, trans_range=0.0, rot_range=0.0, scale_range=0.0, shear_range=0.0, lens_range=0.0):
    """Apply random transformation to the image and mask together
    image: [Height, Width, Channels].
    mask: [Height, Width, Channels]. Same size with image
    
    Returns:
        image': [Height, Width, Channels] tensor. Transformed image.
        mask': [Height, Width, Channels] tensor. Transformed image.
    """
    
    trans = tf.random.normal([2,], mean=0.0, stddev=trans_range)
    rot = tf.random.normal([1,], mean=0.0, stddev=rot_range)
    scale = tf.random.normal([2,], mean=1.0, stddev=scale_range)
    shear = tf.random.normal([2,], mean=0.0, stddev=shear_range)
    lens = tf.random.normal([1,], mean=0.0, stddev=lens_range)
    
#    trans = np.random.normal(0.0, 10, size=2)
#    rot = np.random.normal(0.0, np.pi/24)
#    scale = np.random.normal(1.0, 0.1, size=2)
#    shear = np.random.normal(0.0, 0.05, size=2)
#    lens = np.random.normal(0.0, 0.1)
#    
    image = lens_distortion(image, lens)
    mask = lens_distortion(mask, lens)
    
    # TODO(stephen-baek): Make it accept tf tensor directly, instead of the tuples
    image = affine_transform(image, trans=trans, rot=rot, scale=scale, shear=shear)
    mask = affine_transform(mask, trans=trans, rot=rot, scale=scale, shear=shear)
#    
    return image, mask

# TODO: Validation set: Turn off augmentation
# TODO: Data download needs to be tested
def montgomery(shuffle=True, size=(256,256), trans_range=10.0, rot_range=np.pi/24, scale_range=0.1, shear_range=0.05, lens_range=0.1):
    # If no data folder is found, download the data
    # TODO: Modularize things
    if not os.path.isdir(os.path.join(*[os.getcwd(),'data','montgomery'])):
        data_root_orig = tf.keras.utils.get_file('montgomery.zip',
                                                 'http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip',
                                                 extract=True,
                                                 cache_subdir=os.path.join(*[os.getcwd(),'data']))
        # Remove the zip file
        try:
            os.remove(os.path.join(*['data', 'montgomery.zip']))
        except OSError as e:
            print('dataset.montgomery: ', e)
        # Clean up unnecessary stuff that comes with the ZIP file
        try:
            shutil.rmtree(os.path.join(*['data', '__MACOSX']))
        except OSError as e:
            print('dataset.montgomery: ', e)
        # rename folders
        try:
            from_dir = os.path.join(*['data', 'MontgomerySet'])
            to_dir = os.path.join(*['data', 'montgomery'])
            os.mkdir(to_dir)
            for file in os.listdir(from_dir):
                shutil.move(os.path.join(from_dir, file), to_dir)
            os.rmdir(from_dir)
        except OSError as e:
            print('dataset.montgomery: ', e)
        try:
            from_dir = os.path.join(*['data', 'montgomery', 'ClinicalReadings'])
            to_dir = os.path.join(*['data', 'montgomery', 'clinical_readings'])
            os.mkdir(to_dir)
            for file in os.listdir(from_dir):
                shutil.move(os.path.join(from_dir, file), to_dir)  
            os.rmdir(from_dir)
        except OSError as e:
            print('dataset.montgomery: ', e)
        try:
            from_dir = os.path.join(*['data', 'montgomery', 'CXR_png'])
            to_dir = os.path.join(*['data', 'montgomery', 'images'])
            os.mkdir(to_dir)
            for file in os.listdir(from_dir):
                shutil.move(os.path.join(from_dir, file), to_dir)  
            os.rmdir(from_dir)
        except OSError as e:
            print('dataset.montgomery: ', e)
        try:
            from_dir = os.path.join(*['data', 'montgomery', 'ManualMask'])
            to_dir = os.path.join(*['data', 'montgomery', 'masks'])
            os.mkdir(to_dir)
            for file in os.listdir(from_dir):
                shutil.move(os.path.join(from_dir, file), to_dir)
            os.rmdir(from_dir)
        except OSError as e:
            print('dataset.montgomery: ', e)
        try:
            os.rename(os.path.join(*['data', 'montgomery', 'NLM-MontgomeryCXRSet-ReadMe.pdf']),
            os.path.join(*['data', 'montgomery', 'README.pdf']))
        except OSError as e:
            print('dataset.montgomery: ', e)
            
    data_root = pathlib.Path('./data/montgomery')
    # read image paths using glob and convert them into string format
    image_paths = [str(path) for path in list(data_root.glob('images/*.png'))]
    mask_left_paths = [str(path) for path in list(data_root.glob('masks/left/*.png'))]
    mask_right_paths = [str(path) for path in list(data_root.glob('masks/right/*.png'))]
    count = len(image_paths)
    
    # shuffle them in a random order
    if shuffle:
        temp = list(zip(image_paths, mask_left_paths, mask_right_paths)) # zip the three paths to make sure they are shuffled together
        random.shuffle(temp) # shuffle
        image_paths, mask_left_paths, mask_right_paths = zip(*temp) # unzip them
        image_paths = list(image_paths)           # unzipping converts the lists to tuples for some reason...
        mask_left_paths = list(mask_left_paths)   # we are explicitly converting them back to lists
        mask_right_paths = list(mask_right_paths) #

    # Convert paths to tf.data.Dataset format
    image_path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    mask_left_path_ds = tf.data.Dataset.from_tensor_slices(mask_left_paths)
    mask_right_path_ds = tf.data.Dataset.from_tensor_slices(mask_right_paths)
    
    # Map path datasets to image datasets
    image_ds = image_path_ds.map(lambda x: load_image(x, channels=1, size=size), num_parallel_calls=AUTOTUNE)
    mask_left_ds = mask_left_path_ds.map(lambda x: load_image(x, channels=1, size=size), num_parallel_calls=AUTOTUNE)
    mask_right_ds = mask_right_path_ds.map(lambda x: load_image(x, channels=1, size=size), num_parallel_calls=AUTOTUNE)
    
    # Merge left and right masks
    mask_ds = tf.data.Dataset.zip((mask_left_ds, mask_right_ds))
    mask_ds = mask_ds.map(_mask_merge, num_parallel_calls=AUTOTUNE)
    
    # TODO: Random Noise
    
    # (image, mask) pair
    ds = tf.data.Dataset.zip((image_ds, mask_ds))

    # TODO: Random Transformation
    ds = ds.map(lambda x,y: random_transform(x, y, trans_range=trans_range, rot_range=rot_range, scale_range=scale_range, shear_range=shear_range, lens_range=lens_range), num_parallel_calls=AUTOTUNE)
    
    return ds, count