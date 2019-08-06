import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime

from data import datasets
from models.unet import UNet

tf.compat.v1.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

WIDTH = 256
HEIGHT = 256

montgomery_ds, montgomery_count = datasets.montgomery(size=(HEIGHT, WIDTH))

# Examine
for n, pair in enumerate(montgomery_ds.take(4)):
    image, mask = pair
#    plt.imshow(tf.squeeze(image), cmap='gray')
    plt.imshow(tf.concat([tf.clip_by_value(image+mask*0.1, 0.0, 1.0), image, image], axis=2))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
   
model = UNet()
model.build(input_shape=(None,HEIGHT,WIDTH,1))
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

