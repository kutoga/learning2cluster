import os
import itertools
import shutil

input_dir = './SUN397/SUN397/'
output_dir = 'img_out'

def try_makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# Find all classes
def get_dirs(top_dir):
    for sub_dir in sorted(filter(lambda sub_dir: os.path.isdir(os.path.join(top_dir, sub_dir)), os.listdir(top_dir))):
        yield {
            'name': sub_dir,
            'sub': list(get_dirs(os.path.join(top_dir, sub_dir)))
        }
def get_classes(top_dir):
    def dirs_to_classes(dir, top_path, class_name=''):
        curr_class_name = dir['name'] if class_name == '' else '{}_{}'.format(class_name, dir['name'])
        curr_path = os.path.join(top_path, dir['name'])
        if len(dir['sub']) == 0:
            return [(curr_class_name, curr_path)]
        else:
            return list(itertools.chain(
                *list(map(lambda sub_dir: dirs_to_classes(sub_dir, curr_path, curr_class_name), dir['sub']))
            ))
    return dict(list(itertools.chain(
        *list(map(lambda dir: dirs_to_classes(dir, top_dir), get_dirs(top_dir)))
    )))
classes = get_classes(input_dir)

# Create now the target directory
print("Found {} classes...".format(len(classes)))
try_makedirs(output_dir)
for class_name, class_source_dir in classes.items():
    target_dir = os.path.join(output_dir, class_name)
    print("Copy {} to {}...".format(class_source_dir, target_dir))
    shutil.copytree(class_source_dir, target_dir)
