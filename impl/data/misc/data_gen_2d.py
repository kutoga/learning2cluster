# TODO: This code is imported from another project (VT2); it should be cleaned up

from random import Random
import numpy as np
import matplotlib.pyplot as plt

class DataGen2d:
    def __init__(self):
        self.limit_x = (0, 1)
        self.limit_y = (0, 1)
        self.embedding_dimension = 2
        self.add_noise = True
        self.add_noise_to_data_dimensions = False

    def generate(self, cluster_count=None, records=50, max_dist=0.2, append_mirrored_versions=False):
        rand = Random()
        if cluster_count is None:
            cluster_count = rand.randint(1, 10)
        limits = np.asarray([self.limit_x, self.limit_y])
        limits_scale = np.transpose(np.asarray([[self.limit_x[1] - self.limit_x[0], 0], [0,  self.limit_y[1] - self.limit_y[0]]]))
        limits_offset = np.asarray([self.limit_x[0], self.limit_y[0]])

        rand_points_n = cluster_count + records
        rand_points = np.dot(np.random.rand(rand_points_n, 2), limits_scale) + limits_offset
        rand_point_i = 0

        all_points = np.ones((records + cluster_count, 3), dtype=np.float32) * -1

        def euclidean_norm(p):
            return np.sqrt(np.sum(np.square(p)))

        def distance(p1, p2):
            return euclidean_norm(p1 - p2)

        # Create some clusters
        clusters = []
        for i in range(cluster_count):
            p = rand_points[i]
            clusters.append({
                'data': [p]
            })
        # clusters = [{'data': [rand_points[i]]} for i in range(cluster_count)]

        rand_point_i += cluster_count

        for r in range(records):

            # Choose a cluster
            cluster = rand.choice(clusters)

            # Choose a point from the current cluster
            p0 = rand.choice(cluster['data'])

            # Choose a direction for the new point
            v = 2*(rand_points[rand_point_i] - 0.5)
            rand_point_i += 1

            # v has a maximum length: It should not go outside of our limiting rectangle
            # max{x_f}(
            #  p[0] + x_f * v[0] = limit_x[0],
            #  p[0] + x_f * v[0] = limit_x[1]
            # )
            # max(
            #  (limit_x[0] - p[0]) / v[0],
            #  (limit_x[1] - p[0]) / v[0],
            # )

            x_f = np.max((limits[0] - p0[0]) / v[0])
            y_f = np.max((limits[1] - p0[0]) / v[1])
            v *= min(x_f, y_f)

            # Now we can define a maximum length for v: It is 45% of the distance to any point of another cluster
            other_cluster_points = []
            for c in clusters:
                if c is cluster:
                    continue
                other_cluster_points += c['data']
            other_cluster_points_distances = list(map(
                lambda p: distance(p0, p),
                other_cluster_points
            ))
            v_norm = euclidean_norm(v)
            if len(other_cluster_points_distances) > 0:
                min_dist = min(other_cluster_points_distances)
                max_allowed_distance = min_dist * 0.4

                # Shrink v if required
                if v_norm > max_allowed_distance:
                    v *= max_allowed_distance / v_norm
                    v_norm = max_allowed_distance

            if v_norm > max_dist:
                v *= max_dist / v_norm
                v_norm = max_dist

            # Stretch v with a random factor in [0, 1]
            v *= rand.random()

            cluster['data'].append(p0 + v)

        clusters = list(map(lambda c: c['data'][1:], clusters))
        clusters = list(filter(lambda c: len(c) > 0, clusters))

        if append_mirrored_versions:
            c_x_m = []
            c_y_m = []
            c_xy_m = []

            # Create 3 mirrored versions for each cluster
            for cluster in clusters:
                x_m = []
                y_m = []
                xy_m = []
                for p in cluster:
                    x_m_p = np.copy(p)
                    y_m_p = np.copy(p)
                    xy_m_p = np.copy(p)
                    x_m_p[0] = limits[0][1] - (x_m_p[0] - limits[0][0])
                    y_m_p[1] = limits[1][1] - (y_m_p[1] - limits[1][0])
                    xy_m_p[0] = x_m_p[0]
                    xy_m_p[1] = y_m_p[1]
                    x_m.append(x_m_p)
                    y_m.append(y_m_p)
                    xy_m.append(xy_m_p)
                c_x_m.append(x_m)
                c_y_m.append(y_m)
                c_xy_m.append(xy_m)

            mirrored_clusters = [c_x_m, c_y_m, c_xy_m]

            return clusters, mirrored_clusters
        return clusters

class DataGen2dv02:
    def __init__(self):
        self.limit_x = (0, 1)
        self.limit_y = (0, 1)
        self.embedding_dimension = 2
        self.add_noise = True
        self.add_noise_to_data_dimensions = False
        self.__rand = Random()

    def generate(self, cluster_count=None, records=50, sigma=0.025, append_mirrored_versions=False, cluster_count_min=1, cluster_count_max=10):
        rand = self.__rand

        if cluster_count is None:
            cluster_count = rand.randint(cluster_count_min, cluster_count_max)
        limits = np.asarray([self.limit_x, self.limit_y])
        limits_scale = np.transpose(np.asarray([[self.limit_x[1] - self.limit_x[0], 0], [0,  self.limit_y[1] - self.limit_y[0]]]))
        limits_offset = np.asarray([self.limit_x[0], self.limit_y[0]])

        # Generate some candidates
        possible_cluster_centers = np.dot(np.random.rand(cluster_count * 10, 2), limits_scale) + limits_offset

        # Search now some, but they must have a good distance of at least min_center_dist
        min_center_dist = 0.25
        min_center_dist_squared = min_center_dist * min_center_dist
        cluster_centers = []
        for c_i in range(possible_cluster_centers.shape[0]):
            center = possible_cluster_centers[c_i]
            if len(cluster_centers) > 0:
                min_dist = min(map(lambda c: np.sum(np.square(center - c)), cluster_centers))
                if min_dist >= min_center_dist_squared:
                    cluster_centers.append(center)
            else:
                cluster_centers.append(center)
            if len(cluster_centers) >= cluster_count:
                break
        cluster_count = len(cluster_centers)
        cluster_centers = np.asarray(cluster_centers)
        # cluster_centers = np.dot(np.random.rand(cluster_count, 2), limits_scale) + limits_offset


        # TODO: Find better centers
        data_points_per_cluster = np.random.multinomial(records, [1./cluster_count]*cluster_count)

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








