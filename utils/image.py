"""
Image Data Input Pipeline
Functions to facilitate loading image batches using tf.data
Copyright (c) 2019 Visual Intelligence Laboratory
Licensed under the MIT License (see LICENSE for details)
Written by Stephen Baek
"""

import tensorflow as tf
import numpy as np

def load_image(filepath, channels=3, size=(256,256)):
    """Load an image from a file path.
    filepath: string containing the file path of the image
    
    Returns:
        image: [Height, Width, Channels] tensor.
    """
    # Read raw image (string of pixel values)
    image = tf.io.read_file(filepath)
    
    # Decode the raw image into pixel array
#     image = tf.image.decode_image(image)    # This is supposed to do the same thing as below, but there seems to be a bug in TensorFlow...
    image = tf.cond(tf.image.is_jpeg(image),
      lambda: tf.image.decode_jpeg(image, channels),
      lambda: tf.image.decode_png(image, channels))
    
#     image = tf.image.decode_image(image)   # Decode the raw image into pixel array
    image = tf.cast(image, tf.float32)     # Make sure the pixel values are in floating point numbers, NOT INTEGERS
    image = tf.image.resize(image, [size[0], size[1]])
    image /= 255.0
    
    return image

# TODO(stephen-baek): Can user set the distortion parameters?
def random_transform(image, trans_range=0.0, rot_range=0.0, scale_range=0.0, shear_range=0.0, lens_range=0.0):
    """Apply random transformation to the image
    image: [Height, Width, Channels].
    
    Returns:
        M: [Height, Width, Channels] tensor. Transformed image.
    """
    trans = tf.random.normal([2,], mean=0.0, stddev=trans_range)
    rot = tf.random.normal([1,], mean=0.0, stddev=rot_range)
    scale = tf.random.normal([2,], mean=1.0, stddev=scale_range)
    shear = tf.random.normal([2,], mean=0.0, stddev=shear_range)
    lens = tf.random.normal([1,], mean=0.0, stddev=lens_range)
    
#    image = lens_distortion(image, lens)
    # TODO(stephen-baek): Make it accept tf tensor directly, instead of the tuples
    image = affine_transform(image, trans=trans, rot=rot, scale=scale, shear=shear)
    
    return image

def to_affine_transform_matrix(origin=(0.0, 0.0), trans=(0.0, 0.0), rot=0.0, scale=(1.0, 1.0), shear=(0.0, 0.0)):
    """Create a 3x3 affine transformation matrix from transformation parameters.
    The transformation is applied in the following order: Shear - Scale - Rotate - Translate 
    origin: (x, y). Transformation will take place centered around this pixel location.
    trans: (tx, ty). Translation vector
    rot: theta. Rotation angle
    scale: (sx, sy). Scale (zoom) in x and y directions.
    Shear: (hx, hy). Shear in x and y directions.
    
    Returns:
        M: [3, 3] tensor.
    """
    # Rotation matrix
    #     R = [[ cos(theta)  -sin(theta)   0 ]
    #          [ sin(theta)   cos(theta)   0 ]
    #          [     0            0        1 ]]
    R = tf.convert_to_tensor([[tf.cos(rot), -tf.sin(rot), 0], [tf.sin(rot), tf.cos(rot), 0], [0, 0, 1]], tf.float32)
  
    # Scale and shear
    #         [[ sx  0   0 ]    [[ 1   hx  0 ]    [[  sx   sx*hx   0 ]
    #     S =  [ 0   sy  0 ]  *  [ hy  1   0 ]  =  [ sy*hy   sy    0 ]
    #          [ 0   0   1 ]]    [ 0   0   1 ]]    [   0      0    1 ]]
    S = tf.convert_to_tensor([[scale[0], scale[0]*shear[0], 0], [scale[1]*shear[1], scale[1], 0], [0, 0, 1]], tf.float32)

    # Coordinate transform: shifting the origin from (0,0) to (x, y)
    #     T = [[ 1   0  -x ]
    #          [ 0   1  -y ]
    #          [ 0   0   1 ]]
    M = tf.convert_to_tensor([[1, 0, -origin[0]], [0, 1, -origin[1]], [0, 0, 1]], tf.float32)
    
    # Translation matrix + shift the origin back to (0,0)
    #     T = [[ 1   0   tx + x ]
    #          [ 0   1   ty + y ]
    #          [ 0   0      1   ]]
    T = tf.convert_to_tensor([[1, 0, trans[0]+origin[0]], [0, 1, trans[1]+origin[1]], [0, 0, 1]], tf.float32)
  
    # Combine transformations
    M = tf.matmul(S, M)
    M = tf.matmul(R, M)
    M = tf.matmul(T, M)
  
    return M

# TODO: Implement bilinear interpolation
# TODO: Validity check on the inputs
def map_image(image, map_from, interpolation='Nearest'):
    """Pulls back pixel values of `image` from locations specified by `map_from`.
    new_image[:, :] = image[map_from[:,1], map_from[:,0]]
    
    Inputs:
        image: [Height, Width, Channels].
        map_from: [Height, Width, 2]. x, y coordinate values which the pixel values will be mapped from.
        interpolation: Pixel interpolation method ('Nearest': Nearest neighbor, 'Bilinear': Bilinear interpolation)
                       Current version only supports 'Nearest'
    
    Returns: 
        new_image: [Height, Width, Channels]. A new image of the same size as the input.
    """
    height, width, channels = image.shape
    to_height, to_width = (map_from.shape[0], map_from.shape[1])
    
    flattened = tf.reshape(image, [height*width, channels])  # flatten the input
    
    # We'll create a dummy pixel at the end of the flattened image and make
    # the out of bounds pixels refer to the dummy pixel.
    out_of_bounds = tf.cast(flattened.shape[0], tf.int64)  # pointer to the dummy pixel
    flattened = tf.concat([flattened,  tf.zeros([1,channels], flattened.dtype)], axis=0)
    
    # Flatten the map_from tensor in the same way and split x, y coords.
    map_from = tf.reshape(map_from, [to_height*to_width, 2])
    map_from_x, map_from_y = tf.unstack(map_from, axis=1)
    
    # Interpolation
    if interpolation=='Nearest':
        interp_x = tf.cast(tf.math.round(map_from_x), tf.int64)
        interp_y = tf.cast(tf.math.round(map_from_y), tf.int64)
    elif interpolation=='Bilinear':
        raise Exception('Current version of `map_image` does not support the option `Bilinear`. Use `Nearest` instead.')
    else:
        raise ValueError('Unknown interpolation method: %s'%interpolation)
    
    # Coordinates to indices. Assign the dummy pixel pointer to out of bounds pixels
    ind = interp_y*width + interp_x
    out_of_bounds = tf.tile([out_of_bounds], [tf.cast(ind.shape[0], tf.int64)])  # tf.where does not support broadcasting
    ind = tf.where(tf.less(interp_x, 0), out_of_bounds, ind)
    ind = tf.where(tf.less(interp_y, 0), out_of_bounds, ind)
    ind = tf.where(tf.greater_equal(interp_x, width), out_of_bounds, ind)
    ind = tf.where(tf.greater_equal(interp_y, height), out_of_bounds, ind)
    
    # Gather pixels based on the indices and return new image
    mapped = tf.gather(flattened, ind)
    mapped = tf.reshape(mapped, [to_height, to_width, channels])
    
    return mapped
    

# TODO(stephen_baek): test with image aspect ratios other than 1:1
# TODO(stephen_baek): comment & document
def affine_transform(image, trans=(0.0,0.0), rot=0.0, scale=(1.0,1.0), shear=(0.0, 0.0)):
    M = to_affine_transform_matrix(origin=(tf.cast(image.shape[1],tf.float32)*0.5, tf.cast(image.shape[0],tf.float32)*0.5), trans=trans, rot=rot[0], scale=scale, shear=shear)
    Minv = tf.linalg.inv(M)
    
    height, width, channels = image.shape
    grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    grid_h = tf.ones([height, width], grid_x.dtype)
    grid = tf.stack([grid_x, grid_y, grid_h], axis=2)
    
    map_from = tf.matmul(Minv, tf.cast(grid, Minv.dtype), transpose_b=True)
    map_from = tf.transpose(map_from, perm=[0,2,1])
    map_from_x, map_from_y, map_from_h = tf.unstack(map_from, axis=2)
    map_from = tf.stack([map_from_x, map_from_y], axis=2)
    
    return map_image(image, map_from)

# Barrel distortion model (http://sprg.massey.ac.nz/pdfs/2003_IVCNZ_408.pdf, Gribbon et al. "A Real-time FPGA Implementation of a Barrel Distortion Correction Algorithm with Bilinear Interpolation")
# TODO: Bilinear interpolation
# TODO(stephen_baek): test with image aspect ratios other than 1:1
# TODO(stephen_baek): comment & document
def lens_distortion(image, k=0.0):
    height, width, channels = image.shape
    grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    grid_x = tf.cast(grid_x, tf.float32)/tf.cast(width, tf.float32) - 0.5
    grid_y = tf.cast(grid_y, tf.float32)/tf.cast(height, tf.float32) - 0.5
    
    grid = tf.stack([grid_x, grid_y], axis=2)
    
    polar = tf.atan2(grid_y, grid_x)
    
    new_r_squared = tf.reduce_sum(tf.square(grid), axis=2)
    new_r = tf.sqrt(new_r_squared)
    r = new_r*(1 + k*new_r_squared)
    
    map_from = tf.stack([(r*tf.cos(polar) + 0.5)*tf.cast(width,tf.float32), (r*tf.sin(polar)+0.5)*tf.cast(height,tf.float32)], axis=2)
    
    return map_image(image, map_from)
