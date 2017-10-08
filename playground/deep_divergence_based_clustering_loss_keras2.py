from keras.layers import Input
from keras.models import Model

from impl.nn.base.misc.deep_divergence_based_clustering import get_ddbc_loss_function

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

loss, d_a, triuAAt, d_m = get_ddbc_loss_function(x_inputs, s_inputs)

# model = Model(x_inputs + s_inputs, [d_a, triuAAt, d_m, to_keras_tensor(m_qi)])
model = Model(x_inputs + s_inputs, [loss, d_a, triuAAt, d_m])
pred = model.predict(inputs)

print(pred)
print("db2")