# This test is independent of the MT;-)
# This code is really just to play around. It is a try to implement some neural network like thing that only operates
# on boolean values (which is much faster than working with floats).

import numpy as np
np.random.seed(1)

# The complete network only works with 1d data. Currently we only use {0, 1} data
input_size = 2
hidden_layer_sizes = [100]
output_size = 1

# We need to generate some data. et us program an xor-software
x = [
    [0, 0], [0, 1],
    [1, 0], [1, 1]
]
y = [
    0, 1, 1, 0
]

# That's it for now:)
# First make numpy arrays:
x = np.asarray(x)
y = np.asarray(y)
# The output layer is just another "hidden layer"
hidden_layer_sizes.append(output_size)

# Create the required weights: Every layer contains for each input neuron a weight -1, 0 or 1 and also a bias b (b may be any natural number)
weights = []
for i in range(len(hidden_layer_sizes)):

    # Get the number of inputs for the current layer
    if i == 0:
        input_count = input_size
    else:
        input_count = hidden_layer_sizes[i - 1]

    # Create a weight matrix (input_count, output_count) and a bias matrix
    # The weights are random initialized (-1, 0, 1), except for the bias: It is 0

    w = np.random.randint(-3, 4, (input_count, hidden_layer_sizes[i]))
    b = np.zeros((hidden_layer_sizes[i],), dtype=np.int32)

    weights.append((w, b))

def get_rand_value(p):
    if p == 0:
        return 0
    r = np.random.uniform()
    result = 0
    if p < 0:
        if r < (-p):
            result = -1
    else:
        if r < p:
            result = 1
    return result

def get_binary_delta_weights(arr):
    arr = np.copy(arr)
    arr = np.minimum(arr, 1)
    arr = np.maximum(arr, -1)
    if len(arr.shape) == 1:
        for i in range(arr.shape[0]):
            arr[i] = get_rand_value(arr[i])
    elif len(arr.shape) == 2:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                arr[i, j] = get_rand_value(arr[i, j])
    return arr.astype(np.int32)

# Train (we train currently record by record)
temperature = 0.5
for i in range(100):
    wrong = 0
    w, b = weights[-1]

    w_d = np.zeros_like(w, dtype=np.float32)
    b_d = np.zeros_like(b, dtype=np.float32)
    for i in range(x.shape[0]):
        x_i = x[i]
        y_i = y[i]

        # Do the forward pass
        curr_v = x_i
        for j in range(len(hidden_layer_sizes)):
            (w_j, b_j) = weights[j]

            curr_v = np.dot(curr_v, w_j) + b_j

            # Execute the activation:
            curr_v[curr_v >= 0] = 1
            curr_v[curr_v < 0] = 0
        curr_y = curr_v

        # Compare x_i and y_i and backpropagate the error
        d0 = y_i - curr_y

        # For each y value that is excat we backpropagate nothing; for each which is too high we try to decrease the input
        # weights with a probability of temperature and we also try to decrease the bias (by 1) with the same probability.
        # The abs(bias) is equal to (the number of input neurons + 1).

        layer_temperature = temperature

        # Store the probability to increase a value from a previous layer:
        increase_w_p = np.zeros((hidden_layer_sizes[-1],), dtype=np.float32)
        increase_b_p = np.zeros(())

        if np.sum(np.abs(d0)) > 0:
            wrong += 1

        print("temperature={}".format(temperature))
        for j in range(d0.shape[0]):
            dj = d0[j]

            # Try to increase the current weights
            w_d[:, j] += d0[j] * w[:, j] * layer_temperature

            # Try to increase the current bias
            b_d[j] += d0[j] * layer_temperature

            # Backpropagate the error

            # Create a mask for the input weights
            # increase_w_p += d0[j] * w[:, j]

    bdw = get_binary_delta_weights(w_d)
    w += bdw
    w = np.minimum(w, 3)
    w = np.maximum(w, -3)
    bdb = get_binary_delta_weights(b_d)
    b += bdb
    b_max = hidden_layer_sizes[-1] + 1
    b = np.minimum(b, b_max)
    b = np.maximum(b, -b_max)
    print("sum(bdw)={}, sum(bdb)={}".format(np.sum(np.abs(bdw)), np.sum(np.abs(bdb))))

    weights[-1] = (w, b)

    temperature *= 0.95
    if temperature < .5:
        temperature = .5

    print("Error: {}".format(wrong / x.shape[0]))
    print()


