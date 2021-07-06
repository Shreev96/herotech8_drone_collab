# inspired from https://github.com/zuoxingdong/mazelab/blob/236eef5f7c41b86bb784f506fe1a2e0700a2e48f/mazelab/generators/random_shape_maze.py
import numpy as np

from skimage.draw import random_shapes


def random_shape_maze(width, height, max_shapes, max_size, allow_overlap, shape=None):
    x, _ = random_shapes((height, width), max_shapes, max_size=max_size, multichannel=False, shape=shape, allow_overlap=allow_overlap)
    
    x[x == 255] = 0
    x[np.nonzero(x)] = 1
    
    return x
