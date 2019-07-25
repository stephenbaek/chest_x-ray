import tensorflow as tf
from utils.image import load_image
import matplotlib.pyplot as plt
import numpy as np
import pathlib

tf.enable_eager_execution()

path = str(pathlib.Path('../data/montgomery/images/MCUCXR_0001_0.png'))

image = load_image(path)
print(image.shape)
plt.imshow(np.squeeze(image), cmap='gray')
plt.show()
