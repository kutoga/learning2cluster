import numpy as np
from scipy.misc import imread

def calculate_entropy(arr):
    p = np.asarray([
        np.sum(arr == v) for v in np.unique(arr)
    ]) / float(np.prod(arr.shape))
    return -np.sum(p * np.log2(p))

def calculate_img_entropy(img_file):
    img = imread(img_file, mode='L')
    # img = (img / 32).astype(np.uint8)
    return calculate_entropy(img)

print(calculate_entropy(np.asarray([1, 2, 1])))
print(calculate_entropy(np.asarray([1, 2, 1, 2])))


# Low entropy (hopefully)
print(calculate_img_entropy("E:\\tmp\\test\\srv-lab-t-697\\MT\\facescrub_128x128\\Adrianne_León\\b6d58c42ec81b2820a011348c28a11eb8fa6bef8.jpg"))

# Medium entropy
print(calculate_img_entropy("E:\\tmp\\test\\srv-lab-t-697\\MT\\facescrub_128x128\\Adrianne_León\\ec56738870761e15d197c0e2b220e10013bbbfaa.jpg"))

# High entropy
print(calculate_img_entropy("E:\\tmp\\test\\srv-lab-t-697\\MT\\facescrub_128x128\\Adrianne_León\\c4d9327aeb0bf1fcdeae46c782a39e668890eca8.jpg"))
print(calculate_img_entropy("E:\\tmp\\test\\srv-lab-t-697\\MT\\facescrub_128x128\\Adrianne_León\\21e83ddf16d5cfd74158e992364c7ab72204201b.jpg"))
