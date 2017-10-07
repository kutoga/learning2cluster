from keras.layers import Input, Concatenate, Reshape, Lambda
from keras.models import Model
import keras.backend as K

# Build a model that just calculates the deep_divergence_based_clustering_loss
# (this model cannot be trained)

# First define the x vetcors and the softmaxes

# Dummy example: This should be optimal
x0 = [0.0, 0.0, 1.0, 0.0]
s0 = [1.0, 0.0, 0.0]

x1 = [1.0, 0.0, 0.0, 0.0]
s1 = [0.0, 1.0, 0.0]

x2 = [0.0, 1.0, 0.0, 0.0]
s2 = [0.0, 0.0, 1.0]

x = [x0, x1, x2]
s = [s0, s1, s2]

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
    list(map(lambda i: s[i:(i+1)], range(k)))

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
        c = np.dot(a, b) # WTF? Why does this work??? # TODO: Find out
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
for i in range(1, k): # 1..(k-1)
    for j in range(i + 1, k): # 1..k or 1..(k-1): This is not clear for me?
        nominator = dot(t(A[:, :, i:(i+1)]), Km, A[:, :, j:(j+1)])
        denominator = K.sqrt(dot(
            t(A[:, :, i:(i+1)]), Km, A[:, :, i:(i+1)], t(A[:, :, j:(j+1)]), Km, A[:, :, j:(j+1)]
        )) + K.epsilon()
        nominator = Reshape((1,))(nominator)
        denominator = Reshape((1,))(denominator)
        d_a += nominator / denominator
d_a = to_keras_tensor(d_a)

print("d_a=")
print(d_a)
print()







dot(Km, Km)
print("dbg")

model = Model(x_inputs + s_inputs, [d_a])
pred = model.predict(inputs)

print(pred)
print("db2")