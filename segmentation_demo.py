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

# TODO(stephen-baek): Modularize the dataset so that we can read it like 'datasets.montgomery.load_data()'
montgomery_root = pathlib.Path('./data/montgomery')

# read image paths using glob and convert them into string format
montgomery_image_paths = [str(path) for path in list(montgomery_root.glob('images/*.png'))]
montgomery_mask_left_paths = [str(path) for path in list(montgomery_root.glob('masks/left/*.png'))]
montgomery_mask_right_paths = [str(path) for path in list(montgomery_root.glob('masks/right/*.png'))]
montgomery_count = len(montgomery_image_paths)

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

# Examine
for n, pair in enumerate(montgomery_ds.take(4)):
    image, mask = pair
    plt.subplot(2,2,n+1)
    plt.imshow(tf.concat([tf.clip_by_value(image+mask*0.1, 0.0, 1.0), image, image], axis=2))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# TODO(stephen-baek): put the model in a class
input_size=(384,384,1)
inputs = tf.keras.Input(input_size)
conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = tf.keras.layers.Dropout(0.5)(conv4)
pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = tf.keras.layers.Dropout(0.5)(conv5)

up6 = tf.keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = tf.keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv10 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv1)

model = tf.keras.Model(inputs, conv10)

# TODO(stephen-baek): Implement IOU
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# Train-test Split
train_size = int(montgomery_count*0.7)
val_size = int(montgomery_count*0.15)
test_size = int(montgomery_count*0.15)
montgomery_ds = montgomery_ds.shuffle( reshuffle_each_iteration = False, buffer_size=montgomery_count )
train_ds = montgomery_ds.take(train_size)
test_ds = montgomery_ds.skip(train_size)
val_ds = test_ds.skip(val_size)
test_ds = test_ds.take(test_size)

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
BATCH_SIZE=8
train_ds = train_ds.apply( tf.data.experimental.shuffle_and_repeat(buffer_size=train_size) )
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds.batch(BATCH_SIZE)

model.fit(train_ds, epochs=10, steps_per_epoch=train_size/BATCH_SIZE, validation_data=val_ds)

for n, pair in enumerate(test_ds.take(1)):
    image, mask = pair

predicted = model.predict(image)

plt.subplot(1,2,1)
plt.imshow(tf.concat([tf.clip_by_value(image[0]+mask[0]*0.1, 0.0, 1.0), image[0], image[0]], axis=2))
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(tf.concat([tf.clip_by_value(image[0]+predicted[0]*0.5, 0.0, 1.0), image[0], image[0]], axis=2))
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.show()












