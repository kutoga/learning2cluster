from random import Random

import numpy as np

from impl.data.misc.data_gen_2d import DataGen2dv02
from impl.data.misc.helper import rescale_data, rotate_2d

from sklearn.datasets import make_circles, make_moons

class ExtendedDataGen2d:
    def __init__(self, data_gen_2d=None, rand=Random()):
        self.limit_x = (0, 1)
        self.limit_y = (0, 1)
        self.__rescale_border_percentage = 0.25
        self.__rand = rand
        self.__data_gen_2d = data_gen_2d if data_gen_2d is not None else DataGen2dv02(self.__rand)
        self.__cluster_generators = []
        self.__register_default_cluster_generators()

    def __register_cluster_generator(self, cluster_count, min_samples_per_cluster, generator_f, require_even_sample_count=False):
        self.__cluster_generators.append({
            'generator_f': generator_f,
            'cluster_count': cluster_count,
            'min_samples_per_cluster': min_samples_per_cluster,
            'require_even_sample_count': require_even_sample_count
        })

    def __sklearn_cluster_postprocessing(self, data, return_clusters=None, random_rotation=True):
        """
        :param data:
        :param return_clusters: If None all clusters are returned
        :return:
        """
        data_points, cluster_indices = data

        if return_clusters is None:
            clusters = [[] for i in range(np.max(cluster_indices) + 1)]
            c_i_transform = {i: i for i in range(len(clusters))}
        else:
            clusters = [[] for i in range(len(return_clusters))]
            c_i_transform = {}
            c_i_transformed = 0
            for c_i in return_clusters:
                c_i_transform[c_i] = c_i_transformed
                c_i_transformed += 1

        used_p_i = []
        used_c_i = []

        # Because we rotate and rescale the data, we first have to collect al points
        for i in range(data_points.shape[0]):
            c_i = cluster_indices[i]
            if return_clusters is None or c_i in return_clusters:
                used_p_i.append(i)
                used_c_i.append(c_i_transform[c_i])

        # Select / filter the required data points
        data_points = data_points[used_p_i]

        # Rotate the data if required
        if random_rotation:
            data_points = rotate_2d(data_points, self.__rand.uniform(0, np.pi * 2))

        # Define the new scale
        x_range = (self.__rescale_border_percentage, 1 - self.__rescale_border_percentage)
        y_range = x_range

        # Rescale all data points
        data_points = rescale_data(data_points, x_range, y_range)

        # Fill the data in the target data structure
        for i in range(len(used_p_i)):
            clusters[used_c_i[i]].append(data_points[i])

        return clusters

    def __register_default_cluster_generators(self):

        # Register the circle generator for 1 circle
        self.__register_cluster_generator(1, 10,
            lambda n_samples: self.__sklearn_cluster_postprocessing(
                make_circles(n_samples=n_samples * 2, noise=self.__rand.uniform(0, 0.02), factor=self.__rand.uniform(0.1, 0.4)),
                return_clusters=[1]
        ))

        # Register the circle generator for both circles
        self.__register_cluster_generator(2, 5,
            lambda n_samples: self.__sklearn_cluster_postprocessing(
                make_circles(n_samples=n_samples, noise=self.__rand.uniform(0, 0.02), factor=self.__rand.uniform(0.1, 0.4))
        ), require_even_sample_count=True)

        # Register the moon generator for 1 moon
        self.__register_cluster_generator(1, 10,
            lambda n_samples: self.__sklearn_cluster_postprocessing(
                make_moons(n_samples=n_samples * 2, noise=self.__rand.uniform(0, 0.03)),
                return_clusters=[bool(self.__rand.getrandbits(1))]
        ))

        # Register the moon generator for both moon
        self.__register_cluster_generator(2, 5,
            lambda n_samples: self.__sklearn_cluster_postprocessing(
                make_moons(n_samples=n_samples, noise=self.__rand.uniform(0, 0.03))
        ))

        # Register the good old data gen v2. Register it for 1, 2 and 5 clusters
        for c_c in [1, 2, 5]:
            self.__register_cluster_generator(c_c, 1,
                lambda n_samples, c_c=c_c: self.__data_gen_2d.generate(c_c, n_samples)
            )

    def __get_possible_cluster_generator_old(self, cluster_count, samples_per_cluster):
        cluster_generators = list(filter(lambda c:
            c['cluster_count'] <= cluster_count and
            c['min_samples_per_cluster'] <= samples_per_cluster and
            ((not c['require_even_sample_count']) or ((samples_per_cluster * cluster_count) % 2 == 0)), self.__cluster_generators))
        return self.__rand.choice(cluster_generators)

    def __get_possible_cluster_generator(self, cluster_samples):
        cluster_generators = []
        cluster_count = len(cluster_samples)
        for cluster_generator in self.__cluster_generators:
            if cluster_generator['cluster_count'] > cluster_count:
                continue
            if cluster_generator['min_samples_per_cluster'] > min(cluster_samples[:cluster_generator['cluster_count']]):
                continue
            if cluster_generator['require_even_sample_count'] and (sum(cluster_samples[:cluster_generator['cluster_count']]) % 2 == 1):
                continue
            cluster_generators.append(cluster_generator)
        # cluster_generators = list(filter(lambda c:
        #     c['cluster_count'] <= cluster_count and
        #     c['min_samples_per_cluster'] <= samples_per_cluster and
        #     ((not c['require_even_sample_count']) or ((samples_per_cluster * cluster_count) % 2 == 0)), self.__cluster_generators))
        return self.__rand.choice(cluster_generators)

    def generate(self, cluster_count, n_samples, sigma=0.025):
        assert n_samples >= cluster_count

        # Calculate for each cluster how many samples it should have (it might vary inside a "cluster group", but thats ok)
        samples_per_cluster = n_samples // cluster_count
        cluster_samples = [samples_per_cluster] * cluster_count
        cluster_samples[-1] += (n_samples - (samples_per_cluster * cluster_count))

        # Generate some cluster groups
        cluster_groups = []

        used_generators = [] # Just for debugging
        while len(cluster_samples) > 0:
            # cluster_generator = self.__get_possible_cluster_generator(len(cluster_samples), samples_per_cluster)
            cluster_generator = self.__get_possible_cluster_generator(cluster_samples)
            n_clusters = cluster_generator['cluster_count']
            n_samples_gen = sum(cluster_samples[:n_clusters])
            cluster_samples = cluster_samples[n_clusters:]

            # Generate n_cluster with a total of n_samples samples
            cluster_group = cluster_generator['generator_f'](n_samples_gen)
            if len(cluster_group) != n_clusters: # Debugging
                print("shiat")
                cluster_group = cluster_generator['generator_f'](n_samples_gen)
            if sum(map(len, cluster_group)) != n_samples_gen: # Debugging
                print("shiat")
                cluster_group = cluster_generator['generator_f'](n_samples_gen)
            cluster_groups.append(cluster_group)

            used_generators.append(cluster_generator)

        # Now we have some cluster groups. They have to be put on a grid and all points have to be normalized to
        # the range [0, 1]. Check the trivial case where len(cluster_groups) == 1: Then we just can return the first
        # cluster group
        if len(cluster_groups) == 1:
            return cluster_groups[0]

        # Calculate the minimum grid size (n*n)
        n = int(np.ceil(np.sqrt(len(cluster_groups))))

        # The grid size should be flexible: add some empty cells
        width = n + self.__rand.randint(0, n // 2)
        height = n + self.__rand.randint(0, n // 2)

        # Grid indices for the cluster groups
        grid_indices = list(np.random.choice(list(range(width * height)), len(cluster_groups), replace=False))

        # Create now the final cluster list
        clusters = []
        for cg_i in range(len(cluster_groups)):
            cluster_group = cluster_groups[cg_i]
            grid_index = grid_indices[cg_i]

            # Sometimes try to extend the figures size in any direction
            top_index = grid_index - width
            can_expand_top = top_index >= 0 and top_index not in grid_indices
            bottom_index = grid_index + width
            can_expand_bottom = bottom_index < (width * height) and bottom_index not in grid_indices
            right_index = grid_index + 1
            can_expand_right = (right_index % width) > (grid_index % width) and right_index not in grid_indices
            left_index = grid_index - 1
            can_expand_left = (left_index % width) < (grid_index % width) and left_index not in grid_indices
            top_right_index = top_index + 1
            can_expand_top_right = can_expand_top and can_expand_right and top_right_index not in grid_indices
            top_left_index = top_index - 1
            can_expand_top_left = can_expand_top and can_expand_left and top_left_index not in grid_indices
            bottom_right_index = bottom_index + 1
            can_expand_bottom_right = can_expand_bottom and can_expand_right and bottom_right_index not in grid_indices
            bottom_left_index = bottom_index - 1
            can_expand_bottom_left = can_expand_bottom and can_expand_left and bottom_left_index not in grid_indices

            # Save all possible expansions
            possible_expansions = []
            if can_expand_top:
                possible_expansions.append((0, -1, [top_index]))
            if can_expand_bottom:
                possible_expansions.append((0, 1, [bottom_index]))
            if can_expand_right:
                possible_expansions.append((1, 0, [right_index]))
            if can_expand_left:
                possible_expansions.append((-1, 0, [left_index]))
            if can_expand_top_right:
                possible_expansions.append((1, -1, [top_index, top_right_index, right_index]))
            if can_expand_top_left:
                possible_expansions.append((-1, -1, [top_index, top_left_index, left_index]))
            if can_expand_bottom_right:
                possible_expansions.append((1, 1, [bottom_index, bottom_right_index, right_index]))
            if can_expand_bottom_left:
                possible_expansions.append((-1, 1, [bottom_index, bottom_left_index, left_index]))
            if can_expand_top_left and can_expand_top_right:
                possible_expansions.append(((-1, 1), -1, [left_index, top_left_index, top_index, top_right_index, right_index]))
            if can_expand_bottom_left and can_expand_bottom_right:
                possible_expansions.append(((-1, 1), 1, [left_index, bottom_left_index, bottom_index, bottom_right_index, right_index]))
            if can_expand_top_left and can_expand_bottom_left:
                possible_expansions.append((-1, (-1, 1), [top_index, top_left_index, left_index, bottom_left_index, bottom_index]))
            if can_expand_top_right and can_expand_bottom_right:
                possible_expansions.append((1, (-1, 1), [top_index, top_right_index, right_index, bottom_right_index, bottom_index]))

            # With a probability of 66% try to expand in some direction
            expand_direction = (0, 0, [])
            if len(possible_expansions) > 0 and self.__rand.randint(0, 2) > 0:

                # Try to expand
                expand_direction = self.__rand.choice(possible_expansions)

            grid_indices += expand_direction[2]

            # Get the required variables
            x_start = grid_index % width
            x_cells = 1
            if isinstance(expand_direction[0], tuple):
                x_start += expand_direction[0][0]
                x_cells -= expand_direction[0][0]
                x_cells += expand_direction[0][1]
            elif expand_direction[0] < 0:
                x_start += expand_direction[0]
                x_cells -= expand_direction[0]
            elif expand_direction[0] > 0:
                x_cells += expand_direction[0]
            y_start = grid_index // width
            y_cells = 1
            if isinstance(expand_direction[1], tuple):
                y_start += expand_direction[1][0]
                y_cells -= expand_direction[1][0]
                x_cells += expand_direction[1][1]
            elif expand_direction[1] < 0:
                y_start += expand_direction[1]
                y_cells -= expand_direction[1]
            elif expand_direction[1] > 0:
                y_cells += expand_direction[1]

            # Calculate the resize factor and also the offsets
            # x_offset = (grid_index % width) / width
            # y_offset = (grid_index // width) / height
            # x_cells = 1
            # y_cells = 1
            x_offset = x_start / width
            y_offset = y_start / height
            def transform_point(p):
                p[0] *= x_cells / width
                p[1] *= y_cells / height
                p[0] += x_offset
                p[1] += y_offset
                return p

            # Add all clusters
            clusters += [
                list(map(transform_point, cluster)) for cluster in cluster_group
            ]

        return clusters

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    rand = Random()

    seed = None
    rand.seed(seed)
    np.random.seed(seed)

    dg = ExtendedDataGen2d(rand=rand)
    batch = []
    batch_size = 1
    for i in range(batch_size):
        # clusters, mirrored_clusters = dg.generate(records=50, append_mirrored_versions=True)
        clusters = dg.generate(rand.randint(3, 3), 102)
        batch.append(clusters)
        # batch += mirrored_clusters
        print(i)

    fig, ax = plt.subplots()
    for cluster in batch[0]:
        px = np.asarray(list(map(lambda c: c[0], cluster)))
        py = np.asarray(list(map(lambda c: c[1], cluster)))
        ax.scatter(px, py)
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)

    plt.show(block=True)

