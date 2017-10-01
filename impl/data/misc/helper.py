import numpy as np

def rotation_matrix_2d(rad):
    c_r = np.cos(rad)
    s_r = np.sin(rad)
    r_m = np.asarray([[c_r, -s_r], [s_r, c_r]])
    return r_m

def rotate_2d(data, rad):
    return np.transpose(np.dot(rotation_matrix_2d(rad), np.transpose(data)))

def rescale_data(data, x_range, y_range):

    # Get all required values
    new_dx = x_range[1] - x_range[0]
    new_dy = y_range[1] - y_range[0]
    xmin = np.min(data[:, 0])
    xmax = np.max(data[:, 0])
    ymin = np.min(data[:, 1])
    ymax = np.max(data[:, 1])
    old_dx = xmax - xmin
    old_dy = ymax - ymin

    # And the just use simple math
    data[:, 0] = (data[:, 0] - xmin) * new_dx / old_dx + x_range[0]
    data[:, 1] = (data[:, 1] - ymin) * new_dy / old_dy + y_range[0]

    return data

# The MIT License (MIT)
# Copyright (c) 2016 Vladimir Ignatev
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import sys

# See: https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    if count > 0:
        sys.stdout.write('\r')
    sys.stdout.write('[%s] %s%s ...%s' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
    if count == total:
        sys.stdout.write('\n')
        sys.stdout.flush()
