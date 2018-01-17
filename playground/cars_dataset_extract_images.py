import scipy.io as sio
import os
from PIL import Image

img_source_dir = 'cars_train/cars_train/'
annotations_source = 'car_devkit/devkit/cars_train_annos.mat'
img_target_dir = 'img_out'
img_target_resolution = (512, 512)

# Read all data records
annotations = sio.loadmat(annotations_source)['annotations'][0]
class_data = {}
for record in annotations:
    bbox_y1 = int(record[0])
    bbox_x1 = int(record[1])
    bbox_x2 = int(record[2])
    bbox_y2 = int(record[3])
    record_class = int(record[4])
    filename = record[5][0]

    if record_class not in class_data:
        class_data[record_class] = []

    class_data[record_class].append({
        'filename': filename,
        'bbox': {
            'x1': bbox_x1,
            'x2': bbox_x2,
            'y1': bbox_y1,
            'y2': bbox_y2
        }
    })

# Generate all images
def try_makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
imgs_total = sum(map(len, class_data.values()))
img_curr = 0
for class_name in sorted(class_data.keys()):
    class_target_dir = os.path.join(img_target_dir, '{:05d}'.format(class_name))
    sorted_records = sorted(class_data[class_name], key=lambda x: x['filename'])
    try_makedirs(class_target_dir)
    for i in range(len(sorted_records)):
        img_curr += 1
        record = sorted_records[i]
        target_file_path = os.path.join(class_target_dir, '{:04d}.jpeg'.format(i))
        source_file_path = os.path.join(img_source_dir, record['filename'])
        print("Processing ({}/{}) {} -> {}...".format(img_curr, imgs_total, source_file_path, target_file_path))

        img = Image.open(source_file_path)
        # Cut off the image (given the bounding box things)
        img = img.resize(img_target_resolution, Image.BICUBIC)
        img.save(target_file_path)

