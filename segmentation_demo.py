import tensorflow as tf
from utils.image import load_image
import matplotlib.pyplot as plt
import numpy as np
import pathlib

tf.enable_eager_execution()


image = load_image('./data/montgomery/images/MCUCXR_0001_0.png')
print(image.shape)
plt.imshow(np.squeeze(image), cmap='gray')