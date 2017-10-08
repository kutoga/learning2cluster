from keras.layers import Input, Concatenate, Reshape, Lambda
from keras.models import Model
import keras.backend as K

# Build a model that just calculates the deep_divergence_based_clustering_loss
# (this model cannot be trained)

# First define the x vetcors and the softmaxes

x0 = [0.2, 0.8]
s0 = [0.2, 0.6, 0.2]

x1 = [0.1, 0.4]
s1 = [0.1, 0.2, 0.7]

x2 = [0.4, 0.1]
s2 = [0.7, 0.0, 0.3]

x3 = [0.8, 0.2]
s3 = [0.0, 0.9, 0.1]

x = [x0, x1, x2, x3]
s = [s0, s1, s2, s3]


# # Dummy example: This should be optimal
# x0 = [0.0, 0.0, 1.0, 0.0]
# s0 = [1.0, 0.0, 0.0]
#
# x1 = [1.0, 0.0, 0.0, 0.0]
# s1 = [0.0, 1.0, 0.0]
#
# x2 = [0.0, 1.0, 0.0, 0.0]
# s2 = [0.0, 0.0, 1.0]
#
# x = [x0, x1, x2]
# s = [s0, s1, s2]

###### Generate some x and s values #########
xn = 3 # number of x values
xd = 4 # dimensions of x values
sn = 3 # Softmax classes
xrange = (-1, 1)
srange = (-5, 5)

from random import Random
import numpy as np

rand = Random()
rand.seed(1729)

# Generate all x values
x = [[rand.uniform(*xrange) for d in range(xd)] for i in range(xn)]

# Generate all softmax classes
softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
s = [softmax([rand.uniform(*srange) for c in range(sn)]) for i in range(xn)]
###### End: Generate some x and s values #########


# Let us now start with the magic
import numpy as np

x = np.asarray(x, dtype=np.float32)
s = np.asarray(s, dtype=np.float32)

# Store the number of inputs
n = x.shape[0]
assert s.shape[0] == n

# Store the number of clusters
k = s.shape[1]

# Create the input to test the model:
inputs = list(map(lambda i: x[i:(i+1)], range(n))) + \
    list(map(lambda i: s[i:(i+1)], range(n)))

# Build the model
x_inputs = [Input(x.shape[1:]) for i in range(x.shape[0])]
s_inputs = [Input(s.shape[1:]) for i in range(s.shape[0])]

def to_keras_tensor(x):
    # We need a dummy layer: Use x_inputs[0]; it doesnt matter which layer, because its values are not used
    dl = x_inputs[0]
    return Lambda(lambda dl: x)(dl)

# Make a simplified dot product function for n inputs
def dot(*x):
    if len(x) == 1:
        return x[0]
    res = x[0]
    for i in range(1, len(x)):
        a = res
        b = x[i]
        # c = np.dot(a, b) # WTF? Why does this work??? # TODO: Find out
        c = K.batch_dot(a, b)
        res = c #K.dot(res, x[i])
    res = to_keras_tensor(res)
    return res

# Make a short version of the transpose function
t = lambda x: K.permute_dimensions(x, (0, 2, 1))

# Create some functions to build python lists and convert them to keras matrices
def create_py_matrix(n, m, value=0.):
    return [[value] * m for i in range(n)]

def py_matrix_to_keras_matrix(M):
    return Concatenate(axis=1)(list(map(
        lambda x: Concatenate(axis=2)(x),
        M
    )))

def epsilon():
    return 1e-5

#####################
# Build the K matrix (here we have to call it "Km", because K is already used for the keras backend
#####################

# Define a distance measure
d_sigma = 1.
d = lambda x, y: K.exp(-K.sum(K.square(x - y), axis=1) / d_sigma**2)

Km = create_py_matrix(n, n)
for i in range(n):
    for j in range(n):
        Km[i][j] = Reshape((1,1))(d(x_inputs[i], x_inputs[j]))

# Merge the matrix
Km = py_matrix_to_keras_matrix(Km)

#####################
# Build the A matrix
#####################

A = Concatenate(axis=1)(list(map(lambda x: Reshape((1, k))(x), s_inputs)))

#####################
# Calculate d_{\mathram{hid},\alpha}
#####################

# Calculate d_{\mathram{hid},\alpha}
d_a = 0.
for i in range(k - 1): # 1..(k-1)
    for j in range(i + 1, k): # 1..k or 1..(k-1): This is not clear for me?
        nominator = dot(t(A[:, :, i:(i+1)]), Km, A[:, :, j:(j+1)])
        denominator = K.sqrt(dot(
            t(A[:, :, i:(i+1)]), Km, A[:, :, i:(i+1)], t(A[:, :, j:(j+1)]), Km, A[:, :, j:(j+1)]
        )) + epsilon()
        nominator = Reshape((1,))(nominator)
        denominator = Reshape((1,))(denominator)
        d_a += nominator / denominator
d_a /= k
d_a = to_keras_tensor(d_a)

#####################
# Calculate triu(AA^T)
#####################

# triuAAt = np.triu(dot(A, t(A)), 1)
# triuAAt = np.sum(triuAAt)

triuAAt = dot(A, t(A)) * np.triu(np.ones((n, n), dtype=np.float32), 1)
triuAAt = K.sum(triuAAt, axis=(1, 2))
triuAAt = to_keras_tensor(triuAAt)

#####################
# Calculate all m_{q,i} values
#####################

m_qi = create_py_matrix(n, k) # np.zeros((n, k), dtype=np.float32)
units_vectors = np.eye(k, k)

for q in range(n):
    for i in range(k):
        m_qi[q][i] = K.exp(-K.sum(K.square(A[:, q:(q+1), :] - units_vectors[i:(i+1), :]), axis=(1, 2)))
        m_qi[q][i] = Reshape((1, 1))(m_qi[q][i])
m_qi = py_matrix_to_keras_matrix(m_qi)

#####################
# Calculate d_{\mathram{hid},\alpha}
#####################

# Calculate d_{\mathram{hid},\alpha}
d_m = 0.
for i in range(k - 1): # 1..(k-1)
    for j in range(i + 1, k): # 1..k or 1..(k-1): This is not clear for me?
        nominator = dot(t(m_qi[:, :, i:(i+1)]), Km, m_qi[:, :, j:(j+1)])
        denominator = K.sqrt(dot(
            t(m_qi[:, :, i:(i+1)]), Km, m_qi[:, :, i:(i+1)], t(m_qi[:, :, j:(j+1)]), Km, m_qi[:, :, j:(j+1)]
        )) + epsilon()
        nominator = Reshape((1,))(nominator)
        denominator = Reshape((1,))(denominator)
        d_m += nominator / denominator
d_m /= k
d_m = to_keras_tensor(d_m)

loss = to_keras_tensor(d_a + triuAAt + d_m)



dot(Km, Km)
print("dbg")

# model = Model(x_inputs + s_inputs, [d_a, triuAAt, d_m, to_keras_tensor(m_qi)])
model = Model(x_inputs + s_inputs, [loss, d_a, triuAAt, d_m])
pred = model.predict(inputs)

print(pred)
print("db2")