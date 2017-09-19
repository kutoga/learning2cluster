import numpy as np
import random
import matplotlib.pyplot as plt

def generate_points(n):
    return np.random.randn(n, 2)

# Define the inputs for the algorithm
input_points = [
    (0, 1),
    (1, 0),
    (0.7, 0.3),
    (0.5, 0.5)
]
input_points = generate_points(1000)
initial_clusters = [
    (0.3, 0.7),
    (0.8, 0.2)
]
initial_clusters = generate_points(8)
iterations = 100

# input_points = [
#     [0.97858602, 0.4423193],
#     [0.30963096, 1.0105809],
#     [0.99933183, 0.40783247]
# ]
# initial_clusters = [
#     [0.97858602, 0.4423193],
#     [0.30963096,  1.0105809]
# ]
# iterations = 3
#
# input_points = [
#     (0, 1),
#     (1, 0),
#     (0.7, 0.3),
# ]
# initial_clusters = input_points[:2]
# iterations = 3

input_points = [[[0.432616651058197, 0.9144948720932007], [0.971435010433197, 0.8770499229431152], [0.9573595523834229, 0.8058906197547913], [0.9793320298194885, 0.1249421015381813], [0.19087989628314972, 0.6652524471282959], [0.894627034664154, 0.8206210136413574], [0.6532235145568848, 0.6775975227355957], [0.9727154970169067, 0.8485414385795593], [0.9463914632797241, 0.08141208440065384], [0.9442023634910583, 0.10448088496923447], [0.9106354117393494, 0.7833114266395569], [0.47511160373687744, 0.9113213419914246], [0.9375508427619934, 0.11321943253278732], [0.2106529176235199, 0.6752710938453674], [0.19633497297763824, 0.6862131953239441], [0.6356549859046936, 0.6475337743759155], [0.5130684971809387, 0.921600878238678], [0.44060102105140686, 0.936147153377533], [0.6949571967124939, 0.6504610180854797], [0.9713150262832642, 0.8467308878898621], [0.9318228960037231, 0.10710369050502777], [0.9252308011054993, 0.8424120545387268], [0.22764812409877777, 0.6759783625602722], [0.6592212915420532, 0.6968404650688171], [0.49917593598365784, 0.9417561292648315], [0.9154833555221558, 0.07208676636219025], [1.0019121170043945, 0.07027319818735123], [0.6696194410324097, 0.6719698905944824], [0.20236746966838837, 0.7553068995475769], [0.9376367926597595, 0.03827250748872757], [0.9688427448272705, 0.8101586103439331], [0.9405319690704346, 0.08660855144262314], [0.9473019242286682, 0.08818668127059937], [0.20290321111679077, 0.712782084941864], [0.6579391360282898, 0.6735479831695557], [0.9268555045127869, 0.037088148295879364], [0.6238722205162048, 0.6769453287124634], [0.9344056248664856, 0.8167702555656433], [0.6570436358451843, 0.6719744801521301], [0.44102272391319275, 0.9737398624420166]], [[0.048961102962493896, 0.29342806339263916], [0.465868204832077, 0.2639879286289215], [0.2433384656906128, 0.8356947898864746], [0.6118663549423218, 0.1556873768568039], [0.2336883842945099, 0.8514236211776733], [0.45163339376449585, 0.22674566507339478], [0.23950937390327454, 0.8370385766029358], [0.018517060205340385, 0.2538245618343353], [0.20958596467971802, 0.7992538809776306], [0.22021257877349854, 0.8333390355110168], [0.6164257526397705, 0.19272135198116302], [0.005468070972710848, 0.28895097970962524], [0.6600024700164795, 0.1757567822933197], [0.645350992679596, 0.16697728633880615], [0.6310195326805115, 0.17679844796657562], [0.2163165956735611, 0.8019958734512329], [0.4689295291900635, 0.6703370809555054], [0.45404505729675293, 0.649483323097229], [0.6202943921089172, 0.2426331341266632], [0.41480448842048645, 0.1949690878391266], [0.5213098526000977, 0.73983234167099], [0.2321082502603531, 0.8421519994735718], [0.4798850119113922, 0.22292914986610413], [0.42951658368110657, 0.285061240196228], [0.20580027997493744, 0.8432337641716003], [0.022541066631674767, 0.2892634868621826], [0.03767901286482811, 0.2638035714626312], [0.6321319937705994, 0.18756450712680817], [0.45804762840270996, 0.7213136553764343], [0.02693544700741768, 0.2427418828010559], [0.19431297481060028, 0.8134459853172302], [0.4866158366203308, 0.6903499960899353], [0.433564692735672, 0.23654663562774658], [-0.005068006459623575, 0.29521965980529785], [0.030497994273900986, 0.2972189486026764], [0.47816139459609985, 0.6700353622436523], [0.47868233919143677, 0.6804283261299133], [0.6493542194366455, 0.1846194565296173], [0.2484969198703766, 0.8560168147087097], [0.4396395981311798, 0.2188953012228012]]][0]
initial_clusters = input_points[:5]
initial_clusters = generate_points(5)
iterations = 10


# Some helper functions
def relu(x):
    return max(x, 0)


def softmax(values):
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


def euclidean_distance(x, y, square=False):
    res = np.sum(np.square(np.subtract(x, y)))
    if not square:
        res = np.sqrt(res)
    return res


# Choose more intelligent initial start points
cluster_count = 5
initial_clusters = [input_points[-31]] # The first point is trivial: Just use the first point
k = 2
df = lambda d: sum(d)**k
df = lambda d: np.exp(10*min(d))
# df = lambda d: sum(np.power(d, k))
for i in range(1, cluster_count):
    c = [0, 0]
    s = 0
    weights = []
    for p in input_points:
        dists = [euclidean_distance(cp, p, False) for cp in initial_clusters]
        mind = min(dists)
        d = df(dists)
        c += np.multiply(p, d)
        s += d
        weights.append(d)
    c = np.multiply(c, 1. / s)
    if i == cluster_count - 1:
        print("jaja")
    initial_clusters.append(c)

iterations = 15
# initial_clusters = list(input_points[-5:])
# initial_clusters = generate_points(5)

c_initial_clusters = np.asarray(initial_clusters).tolist()

# Execute the algorithm itself
cluster_centers = initial_clusters
cluster_assignements = [
    [0] * len(initial_clusters) for p in range(len(input_points))
]
def distance(x, y):
    return euclidean_distance(x, y, True)
for itr in range(iterations):
    if itr > 0:
        for c_i in range(len(cluster_centers)):
            c = 0
            s = 0
            for p_i in range(len(input_points)):
                c += np.multiply(cluster_assignements[p_i][c_i], input_points[p_i])
                s += cluster_assignements[p_i][c_i]
            c = list(map(lambda x: x / s, c))
            cluster_centers[c_i] = c

    # Renew
    for p_i in range(len(input_points)):
        p = input_points[p_i]
        for c_i in range(len(cluster_centers)):
            c = cluster_centers[c_i]
            d = distance(p, c)
            print(-(1+3*d)**3)
            cluster_assignements[p_i][c_i] = -(1+3*np.sqrt(d))**3# + 3/(1.+d)
        cluster_assignements[p_i] = softmax(cluster_assignements[p_i])

def print_arr(arr):
    print(np.asarray(arr))

print("Cluster centers:")
print_arr(cluster_centers)
print("Points:")
print_arr(input_points)
print("Cluster assignements:")
print_arr(cluster_assignements)

clusters = [[] for c_i in range(len(initial_clusters))]
for p_i in range(len(input_points)):
    c_i = np.argmax(cluster_assignements[p_i])
    clusters[c_i].append(input_points[p_i])

fig, ax = plt.subplots()
for cluster in clusters:
    px = np.asarray(list(map(lambda c: c[0], cluster)))
    py = np.asarray(list(map(lambda c: c[1], cluster)))
    ax.scatter(px, py, alpha=0.8)
px = np.asarray(list(map(lambda c: c[0], c_initial_clusters)))
py = np.asarray(list(map(lambda c: c[1], c_initial_clusters)))
ax.scatter(px, py, alpha=0.8, color='black')
px = np.asarray(list(map(lambda c: c[0], cluster_centers)))
py = np.asarray(list(map(lambda c: c[1], cluster_centers)))
ax.scatter(px, py, alpha=0.8, color='yellow')


plt.show(block=True)