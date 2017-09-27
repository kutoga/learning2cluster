# TODO: This code is imported from another project (VT2); it should be cleaned up

from random import Random
import numpy as np
import matplotlib.pyplot as plt

class DataGen2dv02:
    def __init__(self, rand=Random()):
        self.limit_x = (0, 1)
        self.limit_y = (0, 1)
        self.embedding_dimension = 2
        self.add_noise = True
        self.add_noise_to_data_dimensions = False
        self.__rand = rand

    def generate(self, cluster_count=None, records=50, sigma=0.025, cluster_count_min=1, cluster_count_max=10, allow_less_clusters=False):
        rand = self.__rand

        if cluster_count is None:
            cluster_count = rand.randint(cluster_count_min, cluster_count_max)

        if records < cluster_count:
            raise Exception("cluster_count must be <= records. cluster_count={}, records={}".format(cluster_count, records))

        limits = np.asarray([self.limit_x, self.limit_y])
        limits_scale = np.transpose(np.asarray([[self.limit_x[1] - self.limit_x[0], 0], [0,  self.limit_y[1] - self.limit_y[0]]]))
        limits_offset = np.asarray([self.limit_x[0], self.limit_y[0]])

        # Generate some candidates
        def generate_centers(count):
            return np.dot(np.random.rand(count, 2), limits_scale) + limits_offset
        additional_cluster_center_count = cluster_count * 10
        possible_cluster_centers = generate_centers(cluster_count + additional_cluster_center_count)

        # Search now some, but they must have a good distance of at least min_center_dist
        min_center_dist = 0.2
        min_center_dist_squared = min_center_dist * min_center_dist
        cluster_centers = []

        while True:

            # Add cluster centers
            for c_i in range(possible_cluster_centers.shape[0]):
                center = possible_cluster_centers[c_i]
                if len(cluster_centers) > 0:

                    # Test for the minimum distance
                    too_short_distance = any(map(
                        lambda c: np.sum(np.square(center - c)) < min_center_dist_squared,
                        cluster_centers
                    ))
                    if not too_short_distance:
                        cluster_centers.append(center)
                else:
                    cluster_centers.append(center)

                if len(cluster_centers) >= cluster_count:
                    break

            # Is the job done?
            if len(cluster_centers) >= cluster_count:
                break

            # Is it required to search more cluster centers
            if allow_less_clusters:
                break

            # Ok, generate new cluster candidates
            possible_cluster_centers = generate_centers(possible_cluster_centers.shape[0])



        cluster_count = len(cluster_centers)
        cluster_centers = np.asarray(cluster_centers)
        # cluster_centers = np.dot(np.random.rand(cluster_count, 2), limits_scale) + limits_offset

        # Each cluster contains at least one elements, therefore we create a distribution for records - cluster_count
        # entries and then we add to every cluster one element
        data_points_per_cluster = np.random.multinomial(records - cluster_count, [1./cluster_count]*cluster_count)
        data_points_per_cluster += 1

        data = np.zeros((records, 3), dtype=np.float32)
        data[:, 1:] = np.random.normal(size=(records, 2), scale=sigma)
        i = 0
        c = 0
        empty_c = 0
        clusters = []
        for dpc in data_points_per_cluster:

            # Set the cluster index
            data[i:(i+dpc), 0] = c - empty_c

            # Add the offset for the cluster center
            data[i:(i+dpc), 1:] += cluster_centers[c]

            if dpc > 0:
                clusters.append([data[i + n, 1:] for n in range(dpc)])
            else:
                empty_c += 1

            c += 1
            i += dpc

        # # DEBUG:
        # if len(clusters) != 10:
        #     print("DEBUG")
        #     print("len: {}".format(len(clusters)))
        #     print(clusters)

        return clusters


        # data[:, 1:] = np.minimum(1, np.maximum(0, ))

        def points_in_range(limit, points):
            return np.minimum(limit[1], np.maximum(limit[0], points))

        def generate_cluster(center, n_points):
            px = points_in_range(limits[0], np.random.normal(center[0], sigma, n_points))
            py = points_in_range(limits[1], np.random.normal(center[1], sigma, n_points))
            p = np.dstack([px, py])[0]
            # return list(map(lambda i: p[i], range(n_points)))
            return [p[i] for i in range(n_points)]

        clusters = []
        for c in range(cluster_count):
            n_points = data_points_per_cluster[c]
            if n_points == 0:
                continue
            clusters.append(generate_cluster(cluster_centers[c], n_points))
        return clusters

if __name__ == '__main__':
    dg = DataGen2dv02()
    batch = []
    batch_size = 200
    for i in range(batch_size):
        # clusters, mirrored_clusters = dg.generate(records=50, append_mirrored_versions=True)
        clusters = dg.generate(records=200, sigma=0.025)
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








