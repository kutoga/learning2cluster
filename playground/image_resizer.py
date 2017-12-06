import glob
import os
from scipy.misc import imsave, imread, imresize

# Config
source_directory = 'R:\\MT\\datasets\\cicc\\data\\output'
new_resolution = (48, 48)
new_file_key = 'CMTBM'
file_type = 'jpg'

# # Find all jpgs
# print("Search all jpgs files...")
# files = glob.glob(os.path.join(source_directory, '**', '*.jpg'))
# print("Found {} files...".format(len(files)))
# print("Sort all filenames...")
# files = sorted(files)
# print("Done...")

# Find all jpg files
processed_files = 0
for subdir, dirs, files in os.walk(source_directory):
    for file in filter(lambda file: file.lower().endswith('.jpg') and not new_file_key in file, files):
        file = os.path.join(source_directory, subdir, file)
        target_file = file + '_{}x{}_{}.{}'.format(new_resolution[0], new_resolution[1], new_file_key, file_type)
        if not os.path.exists(target_file):
            print("{}: Convert {}...".format(processed_files, file))
            img = imread(file, mode="RGB")
            img = imresize(img, new_resolution)
            imsave(target_file, img)
        else:
            print("{}: File is already converted: {}".format(processed_files, file))
        processed_files += 1
print("Done...")
