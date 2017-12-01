import pickle
import glob
import os
from PIL import Image

from keras.layers import Conv2D, Activation, BatchNormalization, Input, MaxPool2D, Dropout, GlobalMaxPool2D, UpSampling2D
from keras.models import Model

import numpy as np
import random

from scipy.misc import imread, imresize

data_dir = './data/'
img_size = (128, 128)

# Load the data (& cache them)
cache_file = os.path.join(data_dir, 'cache_{}x{}.pkl'.format(*img_size))
if not os.path.exists(cache_file):
    def load_img(path):
        print("Load image: {}".format(path))
        img = imread(path)
        img = imresize(img, img_size)
        img = img.astype(np.float32)
        img = 2 * ((img / 255) - 0.5)
        img = np.reshape(img, (1,) + img.shape)
        return img
    get_imgs =  lambda animal: sorted(glob.glob(os.path.join(data_dir, '{}*.jpg'.format(animal))))
    load_imgs = lambda animal: list(map(load_img, get_imgs(animal)))

    # Load all images
    x_cat = np.concatenate(load_imgs('cat'))
    x_dog = np.concatenate(load_imgs('dog'))
    y_cat = np.ones((x_cat.shape[0],), dtype=np.float32)
    y_dog = np.zeros((x_dog.shape[0],), dtype=np.float32)

    # Save them
    with open(cache_file, "wb") as fh:
        pickle.dump(
            ((x_cat, y_cat), (x_dog, y_dog)),
            fh
        )
else:
    print("Load data from cache...")
    with open(cache_file, "rb") as fh:
        ((x_cat, y_cat), (x_dog, y_dog)) = pickle.load(fh)
print("Got {} records with dog images and {} records with cat images...".format(x_dog.shape[0], x_cat.shape[0]))

# Merge the x and y matrices
x_train = np.concatenate((x_cat, x_dog))
y_train = np.concatenate((y_cat, y_dog))

# Build a network
nw_input = Input(img_size + (3,))
nw = nw_input
nw = Dropout(0.5)(nw)

nw = Conv2D(64, (3, 3), padding='same')(nw)
nw = Activation('relu')(nw)
nw = BatchNormalization()(nw)
nw = Conv2D(64, (3, 3), padding='same')(nw)
nw = Activation('relu')(nw)
nw = BatchNormalization()(nw)
nw = MaxPool2D()(nw)
nw = Dropout(0.5)(nw)

nw = Conv2D(128, (3, 3), padding='same')(nw)
nw = Activation('relu')(nw)
nw = BatchNormalization()(nw)
nw = Conv2D(128, (3, 3), padding='same')(nw)
nw = Activation('relu')(nw)
nw = BatchNormalization()(nw)
nw = MaxPool2D()(nw)
nw = Dropout(0.5)(nw)

nw = Conv2D(256, (3, 3), padding='same')(nw)
nw = Activation('relu')(nw)
nw = BatchNormalization()(nw)
nw = Conv2D(256, (3, 3), padding='same')(nw)
nw = Activation('relu')(nw)
nw = BatchNormalization()(nw)
nw = MaxPool2D()(nw)
nw = Dropout(0.5)(nw)

# The "simulated fully connected layers"
# Layer 1
nw = Conv2D(512, (5, 5), padding='same')(nw)
nw = Activation('relu')(nw)
nw = BatchNormalization()(nw)
nw = Dropout(0.5)(nw)
# Layer 2
nw = Conv2D(512, (1, 1), padding='same')(nw)
nw = Activation('relu')(nw)
nw = BatchNormalization()(nw)
nw = Dropout(0.5)(nw)
# Do the binary classification
nw = Conv2D(1, (1, 1), padding='same')(nw)

# This "heatmap" could be extended in way to create a heatmap for every possible class (dont use a softmax: one pixel could be inside of two "possible" candidates). Then
# global pooling is done for each map (=for each class) and these values are then used for a softmax
nw = Activation('sigmoid')(nw)

nw_output_heatmap = UpSampling2D((8, 8), name='output_heatmap')(nw)

nw = GlobalMaxPool2D(name='output_class')(nw)
nw_output_class = nw

model = Model([nw_input], [nw_output_heatmap, nw_output_class])
model.summary()

def save_result_img(base_filename, input_img, output_heatmap, output_probability):
    probability_str = '{0:.6f}'.format(output_probability[0])
    original_img = Image.fromarray(((input_img + 1.0) * 0.5 * 255).astype(np.uint8))
    original_img.save(base_filename + '_input__{}.png'.format(probability_str))
    cat_part_img = Image.fromarray(((input_img + 1.0) * 0.5 * output_heatmap * 255).astype(np.uint8))
    cat_part_img.save(base_filename + '_output_{}.png'.format(probability_str))

def create_examples(output_directory, count):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for i in range(count):
        base_name = os.path.join(output_directory, '{:03d}_'.format(i))
        x_curr_dog = x_dog[random.randint(0, len(x_dog) - 1):][:1]
        x_curr_cat = x_cat[random.randint(0, len(x_cat) - 1):][:1]
        y_curr_dog = model.predict([x_curr_dog])
        y_curr_cat = model.predict([x_curr_cat])
        save_result_img(base_name + 'dog', x_curr_dog[0], y_curr_dog[0][0], y_curr_dog[1][0])
        save_result_img(base_name + 'cat', x_curr_cat[0], y_curr_cat[0][0], y_curr_cat[1][0])

model.compile(
    optimizer='Adam',
    loss={
        'output_class': 'binary_crossentropy'
    },
    metrics={
        'output_class': 'binary_accuracy'
    }
)

for i in range(1000000):
    print("Iteration {}".format(i))
    model.fit([x_train], [y_train], epochs=1, verbose=1, batch_size=128, shuffle=True)
    create_examples('examples/{:05d}_itr_examples'.format(i), 16)

