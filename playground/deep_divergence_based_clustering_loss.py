

# This program contains a numpy based version of the loss function for the deep divergence based clustering paper.
# This code may be used to check a deep learning implementation (value by value).

# Two things are used as input for the algorithm: an embedding and the softmax output.
# Assuming we have 3 softmax classes and an embedding with the dimension 2

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

# Let us now start with the magic
import numpy as np

x = np.asarray(x, dtype=np.float32)
s = np.asarray(s, dtype=np.float32)

# Store the number of inputs
n = x.shape[0]
assert s.shape[0] == n

# Store the number of clusters
k = s.shape[1]

# Make a simplified dot product function for n inputs
def dot(*x):
    if len(x) == 1:
        return x[0]
    res = x[0]
    for i in range(1, len(x)):
        res = np.dot(res, x[i])
    return res

# Make a short version of the transpose function
t = np.transpose

#####################
# Build the K matrix
#####################

# Define a distance measure
d_sigma = 1.
d = lambda x, y: np.exp(-np.sum(np.square(x - y)) / d_sigma**2)

K = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    for j in range(n):
        K[i, j] = d(x[i], x[j])

#####################
# Build the A matrix
#####################

A = np.zeros((n, k), dtype=np.float32)

# Oke;-) we just inefficiently copy the softmaxes;-)
for i in range(n):
    for j in range(k):
        A[i, j] = s[i, j]

#####################
# Calculate d_{\mathram{hid},\alpha}
#####################

# Calculate d_{\mathram{hid},\alpha}
d_a = 0.
for i in range(1, k): # 1..(k-1)
    for j in range(i + 1, k): # 1..k or 1..(k-1): This is not clear for me?
        nominator = dot(t(A[:, i:(i+1)]), K, A[:, j:(j+1)])
        denominator = np.sqrt(dot(
            t(A[:, i:(i+1)]), K, A[:, i:(i+1)], t(A[:, j:(j+1)]), K, A[:, j:(j+1)]
        )) + 1e-15
        d_a += nominator / denominator
d_a /= k

print("d_a=")
print(d_a)
print()

#####################
# Calculate triu(AA^T)
#####################

triuAAt = np.triu(dot(A, t(A)), 1)
triuAAt = np.sum(triuAAt)

print("triuAAt=")
print(triuAAt)
print()

#####################
# Calculate all m_{q,i} values
#####################

m_qi = np.zeros((n, k), dtype=np.float32)

units_vectors = np.eye(k, k)

for q in range(n):
    for i in range(k):
        m_qi[q, i] = np.exp(-np.sum(np.square(A[q] - units_vectors[i])))

#####################
# Calculate d_{\mathram{hid},m}
#####################

# Calculate d_{\mathram{hid},m}
d_m = 0.
for i in range(1, k):  # 1..(k-1)
    for j in range(i + 1, k):  # 1..k or 1..(k-1): This is not clear for me?
        nominator = dot(t(m_qi[:, i:(i+1)]), K, m_qi[:, j:(j+1)])
        denominator = np.sqrt(dot(
            t(m_qi[:, i:(i+1)]), K, m_qi[:, i:(i+1)], t(m_qi[:, j:(j+1)]), K, m_qi[:, j:(j+1)]
        ))
        d_m += nominator / denominator
d_m /= k

print("d_m=")
print(d_m)
print()

#####################
# Print the complete loss
#####################
print("loss=")
print(d_a + triuAAt + d_m)