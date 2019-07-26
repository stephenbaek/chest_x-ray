import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random
import os
from datetime import datetime

from utils.image import load_image
from utils.image import random_transform

from models.unet import UNet

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
    
    
    
model = UNet()
model.build(input_shape=(None,256,256,1))
# TODO(stephen-baek): Implement IOU
model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4), loss='binary_crossentropy', metrics=['accuracy'])

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
BATCH_SIZE=6
train_ds = train_ds.apply( tf.data.experimental.shuffle_and_repeat(buffer_size=train_size) )
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds.batch(BATCH_SIZE)

# Tensorboard stuff
logdir= os.path.join(*['logs', datetime.now().strftime("%Y%m%d-%H%M%S")])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, batch_size=BATCH_SIZE, histogram_freq=1, write_grads=True)

# Early stopping
earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)

# Check point
checkpointdir = os.path.join(*[logdir, 'weights-{epoch:02d}-{val_acc:.4f}.hdf5'])
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpointdir, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

model.fit(train_ds, epochs=1000, steps_per_epoch=train_size/BATCH_SIZE, validation_data=val_ds, callbacks=[tensorboard_callback, earlystopping_callback, checkpoint_callback])
#model.fit(train_ds, epochs=1000, steps_per_epoch=train_size/BATCH_SIZE, validation_data=val_ds)

for n, pair in enumerate(test_ds.take(1)):
    image, mask = pair

predicted = model.predict(image)


i = 3
plt.subplot(1,2,1)
plt.imshow(tf.concat([tf.clip_by_value(image[i]+mask[i]*0.1, 0.0, 1.0), image[i], image[i]], axis=2))
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(tf.concat([tf.clip_by_value(image[i]+predicted[i]*0.5, 0.0, 1.0), image[i], image[i]], axis=2))
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.show()

plt.imshow(tf.squeeze(mask[i]), cmap='gray')
plt.show()
plt.imshow(tf.cast(tf.greater(tf.squeeze(predicted[i]),0.5),tf.float32), cmap='gray')
plt.show()









#
#path = str(pathlib.Path('../data/montgomery/images/MCUCXR_0001_0.png'))
#
#image = load_image(path)
#print(image.shape)
#plt.imshow(np.squeeze(image), cmap='gray')
#plt.show()
