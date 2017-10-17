import numpy as np

sm_prediction = [0., 0.2, 0.4, 0.3, 0.1, 0.]
sm_label =      [0., 0.,  0.,  1.,  0., 0.]

print("Cumulative p: {}".format(np.cumsum(sm_prediction)))
print("Cumulative l: {}".format(np.cumsum(sm_label)))

# Make numpy arrays (and shorter variable names)
p = np.asarray(sm_prediction)
l = np.asarray(sm_label)

# Calculate the thing

# Define a "difference" function. The originally used function is x^2 (but abs(x) is also ok)
# Wished properties of the function:
# df(x) = df(-x)
# df(x) >= 0
df = lambda x: x**2
# df = lambda x: np.abs(x)

d = p - l
loss = np.sum(df(np.cumsum(d))) / d.shape[0]

# Is the value correct? TODO: check this
print("Loss={}".format(loss))

# Expected: (df(0) + df(0.2) + df(0.6) + df(-0.1) + df(0) + df(0)) / 6
# Assuming df(0) == 0: (df(0.2) + df(0.6) + df(-0.1)) / 6
# Assuming df = abs: (0.2 + 0.6 + 0.1) / 6 = 0.15
# Assuming df = x^2: (0.04 + 0.36 + 0.01) / 6 = 0.06833