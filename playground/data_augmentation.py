from scipy.misc import imsave, imread

import Augmentor

p = Augmentor.Pipeline()
# p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
# p.flip_left_right(probability=1.0)
p.flip_top_bottom(probability=1.0)
# p.rotate90(probability=0.5)
# p.rotate270(probability=0.5)

img_in = "C:/Users/bmeier/Documents/Ashampoo Burning Studio 18/data_augmentation_input.jpg"
img_out = "C:/Users/bmeier/Documents/Ashampoo Burning Studio 18/data_augmentation_flip_tb.png"

imsave(img_out, p.sample_with_array(imread(img_in)))

