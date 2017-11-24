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
