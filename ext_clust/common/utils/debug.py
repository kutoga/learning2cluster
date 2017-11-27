"""
    Provides helper methods for debugging.
    Based on previous work of Gygax and Egly.
"""
import numpy as np


def find_duplicates(array1, array2):
    """Finds duplicates in the two given arrays along axis 0.
    """
    for array1_entry in array1:
        for array2_entry in array2:
            if np.array_equal(array1_entry, array2_entry):
                print('found a duplicate')


def find_all_zero_array_entires(array, threshold=5):
    """Finds array_entries with less than threshold non-zero values.
    """
    for i, array_entry in enumerate(array):
        if np.count_nonzero(array_entry) < threshold:
            print('found sample at index {0} with less than {1} non zero values'.format(i, threshold))


def calculate_distribution(map_label, total_speakers):
    """Calculates the distribution of speaker in a batch (map_label of that batch is given).
    """
    distribution = [0. for i in range(total_speakers)]
    for label in map_label:
        distribution[label] += 1

    for i in range(total_speakers):
        distribution[i] /= len(map_label)
    print('Verteilung:')
    print(distribution)
