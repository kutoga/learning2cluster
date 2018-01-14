import pickle
import os
import numpy as np
import gc

from scipy.misc import imsave

# See: https://patrykchrabaszcz.github.io/Imagenet32/
def load_databatch(data_file, output_directory, img_size=64):
    # data_file = os.path.join(data_folder, 'train_data_batch_')

    with open(data_file, 'rb') as fh:
        d = pickle.load(fh)
    x = d['data']
    y = d['labels']
    # mean_image = d['mean']

    # x = x/np.float32(255)
    # mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    # x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)) #.transpose(0, 3, 1, 2)

    # x is now a list of images and y a list of indices
    count = 0
    cnt_cache = {}
    for img, lbl in zip(x, y):
        count += 1

        # save the file
        class_dir = os.path.join(output_directory, '{:05d}'.format(lbl))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        i = 0
        if class_dir in cnt_cache:
            i = cnt_cache[class_dir] + 1
        while True:
            output_file = os.path.join(class_dir, '{:05d}.png'.format(i))
            if os.path.exists(output_file):
                i += 1
            else:
                break
        cnt_cache[class_dir] = i
        print("Save file ({}/{}): {}".format(count, x.shape[0], output_file))
        imsave(output_file, img)

    # # create mirrored images
    # X_train = x[0:data_size, :, :, :]
    # Y_train = y[0:data_size]
    # X_train_flip = X_train[:, :, :, ::-1]
    # Y_train_flip = Y_train
    # X_train = np.concatenate((X_train, X_train_flip), axis=0)
    # Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)
    #
    # return X_train, Y_train

input_file = 'G:/ImageNet64x64/Imagenet64_val/val_data'
# load_databatch(input_file, 'G:/ImageNet64x64/img_files')

def load_file(file):
    load_databatch(file, 'G:/ImageNet64x64/img_files')

# load_file('G:/ImageNet64x64/Imagenet64_val/val_data')
load_file('G:/ImageNet64x64/Imagenet64_train_part1/train_data_batch_1')
# gc.collect()
# load_file('G:/ImageNet64x64/Imagenet64_train_part1/train_data_batch_2')
# gc.collect()
# load_file('G:/ImageNet64x64/Imagenet64_train_part1/train_data_batch_3')
# gc.collect()
# load_file('G:/ImageNet64x64/Imagenet64_train_part1/train_data_batch_4')
# gc.collect()
# load_file('G:/ImageNet64x64/Imagenet64_train_part1/train_data_batch_5')
# gc.collect()
# load_file('G:/ImageNet64x64/Imagenet64_train_part2/train_data_batch_6')
# gc.collect()
# load_file('G:/ImageNet64x64/Imagenet64_train_part2/train_data_batch_7')
# gc.collect()
# load_file('G:/ImageNet64x64/Imagenet64_train_part2/train_data_batch_8')
# gc.collect()
# load_file('G:/ImageNet64x64/Imagenet64_train_part2/train_data_batch_9')
# gc.collect()
# load_file('G:/ImageNet64x64/Imagenet64_train_part2/train_data_batch_10')
# gc.collect()
