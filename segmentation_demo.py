import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random
import os
from datetime import datetime

from utils.image import load_image
from utils.image import random_transform

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE

# TODO: This should be moved to utils?
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
conv10 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = tf.keras.Model(inputs, conv10)

# TODO(stephen-baek): Implement IOU
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

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
# TODO: Use `tf.data.Dataset.shuffle(buffer_size, seed)` followed by `tf.data.Dataset.repeat(count)`. Static tf.data optimizations will take care of using the fused implementation.
BATCH_SIZE=8
train_ds = train_ds.apply( tf.data.experimental.shuffle_and_repeat(buffer_size=train_size) )
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.batch(val_size)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds.batch(test_size)

# Tensorboard stuff
logdir= os.path.join(*['logs', datetime.now().strftime("%Y%m%d-%H%M%S")])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, batch_size=BATCH_SIZE, histogram_freq=1, write_grads=True)

# Early stopping
earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)

# Check point
checkpointdir = os.path.join(*['logs', 'weights', 'weights-{epoch:02d}-{val_acc:.2f}.hdf5'])
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpointdir, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.fit(train_ds, epochs=1000, steps_per_epoch=train_size/BATCH_SIZE, validation_data=val_ds, callbacks=[tensorboard_callback, earlystopping_callback, checkpoint_callback])
#model.fit(train_ds, epochs=1000, steps_per_epoch=train_size/BATCH_SIZE, validation_data=val_ds)

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

i=2
plt.imshow(tf.squeeze(mask[i]), cmap='gray')
plt.show()
plt.imshow(tf.cast(tf.greater(tf.squeeze(predicted[i]),0.5),tf.float32), cmap='gray')
plt.show()











