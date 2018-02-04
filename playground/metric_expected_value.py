import numpy as np

from sklearn import metrics

from core.nn.misc.MR import misclassification_rate

from core.nn.misc.BBN import BBN

def expected_value(fn_metric, clusters, objects, n=10000, min_objects_per_cluster=1):
    from random import randint, shuffle

    def generate_clusters():
        cluster_objs = np.ones((clusters,), dtype=np.int) * min_objects_per_cluster
        count = objects - clusters * min_objects_per_cluster
        while count > 0:
            idx = randint(0, clusters - 1)
            cluster_objs[idx] += 1
            count -= 1

        res = []
        for i in range(clusters):
            res += [i] * cluster_objs[i]
        shuffle(res)

        return res

    mr_sum = 0.
    for i in range(n):
        mr_sum += fn_metric(
            generate_clusters(),
            generate_clusters()
        )
    mr_avg = mr_sum / n
    return mr_avg

metrics = {
    'NMI': metrics.normalized_mutual_info_score,
    'BBN_norm': lambda y_true, y_pred: BBN(y_true, y_pred, Q=0, normalize=True),
    'MR': misclassification_rate
}

n = 1000
clusters = 5
min_cluster_size = 2
objects = 20
for metric_name, fn_metric in metrics.items():
    print()
    print(metric_name)
    e = []
    for i in range(1, clusters + 1):
        ei = expected_value(fn_metric, i, objects, n, min_cluster_size)
        print("E[{}] = {}".format(i, ei))
        e.append(ei)
    print("E[{}] = {}".format(metric_name, np.mean(e)))
    print("sqrt(VAR[{}]) = {}".format(metric_name, np.std(e)))
