"""
Image Data Input Pipeline
Functions to facilitate loading image batches using tf.data
Copyright (c) 2019 Visual Intelligence Laboratory
Licensed under the MIT License (see LICENSE for details)
Written by Stephen Baek
"""

import tensorflow as tf
import numpy as np

# TODO(stephen-baek): make the output size controllable
def load_image(filepath):
    """Load an image from a file path.
    filepath: string containing the file path of the image
    
    Returns:
        image: [Height, Width, Channels] tensor.
    """
    WIDTH = 256
    HEIGHT = 256
    
    # Read raw image (string of pixel values)
    image = tf.io.read_file(filepath)
    
    # Decode the raw image into pixel array
#     image = tf.image.decode_image(image)    # This is supposed to do the same thing as below, but there seems to be a bug in TensorFlow...
    image = tf.cond(tf.image.is_jpeg(image),
      lambda: tf.image.decode_jpeg(image),
      lambda: tf.image.decode_png(image))
    
#     image = tf.image.decode_image(image)   # Decode the raw image into pixel array
    image = tf.cast(image, tf.float32)     # Make sure the pixel values are in floating point numbers, NOT INTEGERS
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image /= 255.0
    
    return image

# TODO(stephen-baek): Can user set the distortion parameters?
def random_transform(image):
    """Apply random transformation to the image
    image: [Height, Width, Channels].
    
    Returns:
        M: [Height, Width, Channels] tensor. Transformed image.
    """
    trans = tf.random.normal([2], mean=0.0, stddev=10)
    rot = tf.random.normal([1], mean=0.0, stddev=np.pi/24)
    scale = tf.random.normal([2], mean=1.0, stddev=0.1)
    shear = tf.random.normal([2], mean=0.0, stddev=0.05)
    lens = tf.random.normal([1], mean=0.0, stddev=0.1)
    
    image = lens_distortion(image, lens)
    # TODO(stephen-baek): Make it accept tf tensor directly, instead of the tuples
    image = affine_transform(image, trans=(trans[0], trans[1]), rot=rot[0], scale=(scale[0], scale[1]), shear=(shear[0], shear[1]))
    
    return image

def _to_affine_transform_matrix(origin=(0.0, 0.0), trans=(0.0, 0.0), rot=0.0, scale=(1.0, 1.0), shear=(0.0, 0.0)):
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
    R = tf.Variable(lambda: tf.zeros((3,3)), tf.float32)
    cos = tf.cast(tf.cos(rot), tf.float32)
    sin = tf.cast(tf.sin(rot), tf.float32)
    tf.scatter_nd_update(R, [[2,2],[0,0],[1,1],[0,1],[1,0]], [1, cos, cos, -sin, sin])
  
    # Scale and shear
    #         [[ sx  0   0 ]    [[ 1   hx  0 ]    [[  sx   sx*hx   0 ]
    #     S =  [ 0   sy  0 ]  *  [ hy  1   0 ]  =  [ sy*hy   sy    0 ]
    #          [ 0   0   1 ]]    [ 0   0   1 ]]    [   0      0    1 ]]
    S = tf.Variable(lambda: tf.zeros((3,3)), tf.float32)
    tf.scatter_nd_update(S, [[2,2],[0,0],[1,1],[0,1],[1,0]], [1, scale[0], scale[1], scale[0]*shear[0], scale[1]*shear[1]])

    # Coordinate transform: shifting the origin from (0,0) to (x, y)
    #     T = [[ 1   0  -x ]
    #          [ 0   1  -y ]
    #          [ 0   0   1 ]]
    M = tf.Variable(lambda: tf.zeros((3,3)), tf.float32)
    tf.scatter_nd_update(M, [[0,0],[1,1],[2,2],[0,2],[1,2]], [1, 1, 1, -origin[0], -origin[1]])
    
    # Translation matrix + shift the origin back to (0,0)
    #     T = [[ 1   0   tx + x ]
    #          [ 0   1   ty + y ]
    #          [ 0   0      1   ]]
    T = tf.Variable(lambda: tf.zeros((3,3)), tf.float32)
    tf.scatter_nd_update(T, [[0,0],[1,1],[2,2],[0,2],[1,2]], [1, 1, 1, trans[0]+origin[0], trans[1]+origin[1]])
  
    # Combine transformations
    M = tf.matmul(S, M)
    M = tf.matmul(R, M)
    M = tf.matmul(T, M)
  
    return M

# TODO(stephen-baek): move to_keep out of this function and make its own.
def image_map(image, src, dst):
    """Maps pixel values from a source image to a destination image via the mapping defined by src and dst.
    image: [Height, Width, Channels].
    src: [N, 2]. x, y coordinate values which the pixel values will be mapped from. Mapping happens at N such positions.
    dst: [N, 2]. x, y coordinate values which the pixel values will be mapped to.
    
    Returns: 
        new_image: [Height, Width, Channels] tensor. A new image of the same size as the input.
                   new_image[dst[:,1], dst[:,2]] = image[src[:,1], src[:,2]]
    """    
    # make sure the values are mapped from valid positions: 0 <= src[:,1] < Width  &&  0 <= src[:,0] < Height
    to_del = np.logical_or(np.less(src, 0), np.greater_equal(src, np.expand_dims([int(image.shape[0]),int(image.shape[1])], axis=0)))
    to_del = np.any(to_del, axis=1)
    to_del = np.squeeze(np.where(to_del))
    src = np.delete(src, to_del, axis=0)
    dst = np.delete(dst, to_del, axis=0)
    
    # new_image[dst[:,1], dst[:,2]] = image[src[:,1], src[:,2]]
#    new_image = np.zeros((image.shape[0],image.shape[1]))
#    new_image[dst[:,1], dst[:,0]] = image[src[:,1], src[:,0]]
#    new_image[dst[:,1],dst[:,0]] = tf.gather_nd(image, src[:,[1,0]]).eval()
#    new_image = tf.Variable(tf.zeros([int(image.shape[0]),int(image.shape[1])]))
#    new_image = tf.scatter_nd_update(new_image, dst[:,[1,0]], tf.gather_nd(image, src[:,[1,0]]))
        
    new_image = tf.Variable(tf.zeros(image.shape))
    
    return new_image

# TODO: Bilinear interpolation
# TODO(stephen_baek): test with image aspect ratios other than 1:1
# TODO(stephen_baek): comment & document
def affine_transform(image, trans=(0.0,0.0), rot=0.0, scale=(1.0,1.0), shear=(0.0, 0.0)):
  M = _to_affine_transform_matrix(origin=(tf.cast(image.shape[1],tf.float32)/2, tf.cast(image.shape[0],tf.float32)/2), trans=trans, rot=rot, scale=scale, shear=shear)
  
  I = tf.tile(tf.expand_dims(tf.range(image.shape[1]), axis=0), [image.shape[1],1])
  J = tf.tile(tf.expand_dims(tf.range(image.shape[0]), axis=1), [1,image.shape[0]])
  I = tf.cast(tf.reshape(I, (-1,1)), tf.float32)
  J = tf.cast(tf.reshape(J, (-1,1)), tf.float32)
  K = tf.expand_dims(tf.ones(I.shape[0]), axis=1)

  new_coords = tf.transpose(tf.concat([I, J, K], axis=1))

  Minv = tf.linalg.inv(M)

  coords = tf.matmul(Minv, new_coords)

  new_coords = tf.cast(new_coords, tf.int32)
  coords = tf.cast(tf.math.round(coords), tf.int32)

  new_coords = tf.transpose(tf.slice(new_coords, [0, 0], [2, -1]))   # drop the homogeneous coordinate
  coords = tf.transpose(tf.slice(coords, [0, 0], [2, -1]))
  
  return image_map(image, coords, new_coords)

# Barrel distortion model (http://sprg.massey.ac.nz/pdfs/2003_IVCNZ_408.pdf, Gribbon et al. "A Real-time FPGA Implementation of a Barrel Distortion Correction Algorithm with Bilinear Interpolation")
# TODO: Bilinear interpolation
# TODO(stephen_baek): test with image aspect ratios other than 1:1
# TODO(stephen_baek): comment & document
def lens_distortion(image, k=0.0):
  w = int(image.shape[1])-1
  h = int(image.shape[0])-1

  I = np.reshape(np.tile(np.expand_dims(np.linspace(-0.5, 0.5, w+1), axis=0), [h+1,1]), (-1,1))
  J = np.reshape(np.tile(np.expand_dims(np.linspace(-0.5, 0.5, h+1), axis=1), [1,w+1]), (-1,1))

  new_coords = np.concatenate((I, J), axis=1)

  polar = np.squeeze(np.arctan2(J, I))
  new_r_squared = np.sum(np.square(new_coords), axis=1)
  new_r = np.sqrt(new_r_squared)
  r = new_r*(1+k*new_r_squared)

  Iu = ((r*np.cos(polar)+0.5)*w).astype('int32')
  Ju = ((r*np.sin(polar)+0.5)*h).astype('int32')
  coords = np.stack([Iu, Ju], axis=1)
  new_coords = (np.around((new_coords + 0.5)*[w,h])).astype('int32')

  return image_map(image, coords, new_coords)
