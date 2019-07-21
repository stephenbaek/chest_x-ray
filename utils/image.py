"""
Image Data Input Pipeline
Functions to facilitate loading image batches using tf.data
Copyright (c) 2019 Visual Intelligence Laboratory
Licensed under the MIT License (see LICENSE for details)
Written by Stephen Baek
"""

import tensorflow as tf

# TODO(stephen-baek): make the output size controllable
def load_image(filepath):
    """Load an image from a file path.
    filepath: string containing the file path of the image
    
    Returns:
        image: [Height, Width, Channels] tensor.
    """
    WIDTH = 384
    HEIGHT = 384
    
    # Read raw image (string of pixel values)
    image = tf.read_file(filepath)
    
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
    R = tf.Variable(tf.zeros((3,3)), tf.float32)
    cos = tf.cast(tf.cos(rot), tf.float32)
    sin = tf.cast(tf.sin(rot), tf.float32)
    tf.scatter_nd_update(R, [[2,2],[0,0],[1,1],[0,1],[1,0]], [1, cos, cos, -sin, sin])
  
    # Scale and shear
    #         [[ sx  0   0 ]    [[ 1   hx  0 ]    [[  sx   sx*hx   0 ]
    #     S =  [ 0   sy  0 ]  *  [ hy  1   0 ]  =  [ sy*hy   sy    0 ]
    #          [ 0   0   1 ]]    [ 0   0   1 ]]    [   0      0    1 ]]
    S = tf.Variable(tf.zeros((3,3)), tf.float32)
    tf.scatter_nd_update(S, [[2,2],[0,0],[1,1],[0,1],[1,0]], [1, scale[0], scale[1], scale[0]*shear[0], scale[1]*shear[1]])

    # Coordinate transform: shifting the origin from (0,0) to (x, y)
    #     T = [[ 1   0  -x ]
    #          [ 0   1  -y ]
    #          [ 0   0   1 ]]
    M = tf.Variable(tf.zeros((3,3)), tf.float32)
    tf.scatter_nd_update(M, [[0,0],[1,1],[2,2],[0,2],[1,2]], [1, 1, 1, -origin[0], -origin[1]])
    
    # Translation matrix + shift the origin back to (0,0)
    #     T = [[ 1   0   tx + x ]
    #          [ 0   1   ty + y ]
    #          [ 0   0      1   ]]
    T = tf.Variable(tf.zeros((3,3)), tf.float32)
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
    to_keep = tf.logical_and(tf.greater_equal(src, 0), tf.less(src, tf.expand_dims(tf.cast([image.shape[1], image.shape[0]], tf.int32), axis=0)))
    to_keep = tf.reduce_all(to_keep, axis=1)
    to_keep = tf.squeeze(tf.where(to_keep))
    src = tf.stack([tf.gather(src[:,0], to_keep), tf.gather(src[:,1], to_keep)])
    dst = tf.stack([tf.gather(dst[:,0], to_keep), tf.gather(dst[:,1], to_keep)])
    src = tf.transpose(src)
    dst = tf.transpose(dst)

    # new_image[dst[:,1], dst[:,2]] = image[src[:,1], src[:,2]]
    new_image = tf.Variable(tf.zeros(image.shape))
    new_image = tf.scatter_nd_update(new_image, tf.reverse(dst, axis=[1]), tf.gather_nd(image, tf.reverse(src,axis=[1])))
  
    return new_image

# TODO: Bilinear interpolation
# TODO(stephen_baek): test with image aspect ratios other than 1:1
# TODO(stephen_baek): comment & document
def affine_transform(image, trans=(0.0,0.0), rot=0.0, scale=(1.0,1.0), shear=(0.0, 0.0)):
  M = to_affine_transform_matrix(origin=(tf.cast(image.shape[1],tf.int32)//2, tf.cast(image.shape[0],tf.int32)//2), trans=trans, rot=rot, scale=scale, shear=shear)
  
  I = tf.tile(tf.expand_dims(tf.range(image.shape[1]), axis=0), [image.shape[1],1])
  J = tf.tile(tf.expand_dims(tf.range(image.shape[0]), axis=1), [1,image.shape[0]])
  I = tf.cast(tf.reshape(I, (-1,1)), tf.float32)
  J = tf.cast(tf.reshape(J, (-1,1)), tf.float32)
  K = tf.expand_dims(tf.ones(I.shape[0]), axis=1)

  new_coords = tf.transpose(tf.concat([I, J, K], axis=1))

  Minv = tf.matrix_inverse(M)

  coords = tf.matmul(Minv, new_coords)

  new_coords = tf.cast(new_coords, tf.int32)
  coords = tf.cast(coords, tf.int32)

  new_coords = tf.transpose(tf.slice(new_coords, [0, 0], [2, -1]))   # drop the homogeneous coordinate
  coords = tf.transpose(tf.slice(coords, [0, 0], [2, -1]))
  
  return image_map(image, coords, new_coords)

# Barrel distortion model (http://sprg.massey.ac.nz/pdfs/2003_IVCNZ_408.pdf, Gribbon et al. "A Real-time FPGA Implementation of a Barrel Distortion Correction Algorithm with Bilinear Interpolation")
# TODO: Bilinear interpolation
# TODO(stephen_baek): test with image aspect ratios other than 1:1
# TODO(stephen_baek): comment & document
def lens_distortion(image, k):
  w = tf.cast(image.shape[1], tf.float32)-1
  h = tf.cast(image.shape[0], tf.float32)-1

  I = tf.reshape(tf.tile(tf.expand_dims(tf.linspace(-0.5, 0.5, image.shape[1]+1), axis=0), [image.shape[1],1]), (-1,1))
  J = tf.reshape(tf.tile(tf.expand_dims(tf.linspace(-0.5, 0.5, image.shape[0]+1), axis=1), [1,image.shape[0]]), (-1,1))

  new_coords = tf.concat([I, J], axis=1)

  polar = tf.atan2(J, I)
  new_r_squared = tf.reduce_sum(tf.square(new_coords), axis=1)
  new_r = tf.sqrt(new_r_squared)
  r = new_r*(1+k*new_r_squared)

  Iu = tf.cast((r*tf.cos(tf.squeeze(polar))+0.5)*w, tf.int32)
  Ju = tf.cast((r*tf.sin(tf.squeeze(polar))+0.5)*h, tf.int32)
  coords = tf.stack([Iu, Ju], axis=1)
  new_coords = tf.cast((new_coords + 0.5)*[w,h], tf.int32)

#   coords = tf.transpose(coords)
#   new_coords = tf.transpose(new_coords)

  return image_map(image, coords, new_coords)
