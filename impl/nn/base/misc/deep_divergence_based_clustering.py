import numpy as np

from keras.layers import Input, Concatenate, Reshape, Lambda
from keras.models import Model
import keras.backend as K

from core.nn.helper import concat


def get_ddbc_loss_function(x_inputs, classification_inputs, alpha=1., beta=1., gamma=1., force_to_keras_tensors=True):
    """
    A keras implementation of the loss function of the paper Deep Divergence-Based Clustering.

    :param x_inputs: All x inputs (a python array)
    :param classification_inputs: All classification inputs (each x-value requires exactly one classification input)
    :param alpha: How much to weight d_{\mathrm{hid},\alpha}, default value is 1.0
    :param beta: How much to weight triu(AA^T)
    :param gamma: How much to weight d_{\mathrm{hid},m}, default value is 1.0
    :return: loss, d_{\mathrm{hid},\alpha}, triu(AA^T), d_{\mathrm{hid},m}
    """
    # x_inputs = [Input(x.shape[1:]) for i in range(x.shape[0])]
    s_inputs = classification_inputs #[Input(s.shape[1:]) for i in range(s.shape[0])]

    # Store the number of inputs
    n = len(x_inputs)
    assert len(s_inputs) == n
    assert n > 0

    # Store the number of clusters
    k = int(s_inputs[0].shape[1])

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
            res = c  # K.dot(res, x[i])
        res = to_keras_tensor(res)
        return res

    # Make a short version of the transpose function
    t = lambda x: K.permute_dimensions(x, (0, 2, 1))

    # Create some functions to build python lists and convert them to keras matrices
    def create_py_matrix(n, m, value=0.):
        return [[value] * m for i in range(n)]

    def py_matrix_to_keras_matrix(M):
        return concat(axis=1, inputs=list(map(
            lambda x: concat(axis=2, inputs=x),
            M
        )))
        # return concat_layer(axis=1, input_count=len(M))(list(map(
        #     lambda x: concat_layer(axis=2, input_count=len(x))(x),
        #     M
        # )))
        # return Concatenate(axis=1)(list(map(
        #     lambda x: Concatenate(axis=2)(x),
        #     M
        # )))

    def epsilon():
        # return K.epsilon()
        return 1e-5

    #####################
    # Build the K matrix (here we have to call it "Km", because K is already used for the keras backend
    #####################

    # Define a distance measure
    d_sigma = 1.
    d = lambda x, y: K.exp(-K.sum(K.square(x - y), axis=1) / d_sigma ** 2)

    Km = create_py_matrix(n, n)
    for i in range(n):
        for j in range(n):
            Km[i][j] = Reshape((1, 1))(d(x_inputs[i], x_inputs[j]))

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
    for i in range(k - 1):  # 1..(k-1)
        for j in range(i + 1, k):  # 1..k or 1..(k-1): This is not clear for me?
            nominator = dot(t(A[:, :, i:(i + 1)]), Km, A[:, :, j:(j + 1)])
            denominator = K.sqrt(dot(
                t(A[:, :, i:(i + 1)]), Km, A[:, :, i:(i + 1)], t(A[:, :, j:(j + 1)]), Km, A[:, :, j:(j + 1)]
            )) + epsilon()
            nominator = Reshape((1,))(nominator)
            denominator = Reshape((1,))(denominator)
            d_a += nominator / denominator
    d_a /= k
    if d_a != 0.:
        d_a = to_keras_tensor(d_a)

    #####################
    # Calculate triu(AA^T)
    #####################

    # triuAAt = np.triu(dot(A, t(A)), 1)
    # triuAAt = np.sum(triuAAt)

    triuAAt = dot(A, t(A)) * np.triu(np.ones((n, n), dtype=np.float32), 1)
    triuAAt = K.sum(triuAAt, axis=(1, 2))
    triuAAt = to_keras_tensor(triuAAt)
    triuAAt = Reshape((1,))(triuAAt)

    #####################
    # Calculate all m_{q,i} values
    #####################

    m_qi = create_py_matrix(n, k)  # np.zeros((n, k), dtype=np.float32)
    units_vectors = np.eye(k, k)

    for q in range(n):
        for i in range(k):
            m_qi[q][i] = K.exp(-K.sum(K.square(A[:, q:(q + 1), :] - units_vectors[i:(i + 1), :]), axis=(1, 2)))
            m_qi[q][i] = Reshape((1, 1))(m_qi[q][i])
    m_qi = py_matrix_to_keras_matrix(m_qi)

    #####################
    # Calculate d_{\mathram{hid},\alpha}
    #####################

    # Calculate d_{\mathram{hid},\alpha}
    d_m = 0.
    for i in range(k - 1):  # 1..(k-1)
        for j in range(i + 1, k):  # 1..k or 1..(k-1): This is not clear for me?
            nominator = dot(t(m_qi[:, :, i:(i + 1)]), Km, m_qi[:, :, j:(j + 1)])
            denominator = K.sqrt(dot(
                t(m_qi[:, :, i:(i + 1)]), Km, m_qi[:, :, i:(i + 1)], t(m_qi[:, :, j:(j + 1)]), Km, m_qi[:, :, j:(j + 1)]
            )) + epsilon()
            nominator = Reshape((1,))(nominator)
            denominator = Reshape((1,))(denominator)
            d_m += nominator / denominator
    d_m /= k
    if d_m != 0.:
        d_m = to_keras_tensor(d_m)

    loss = to_keras_tensor(alpha * d_a + beta * triuAAt + gamma * d_m)

    if force_to_keras_tensors:
        d_a = to_keras_tensor(0. * loss + d_a)
        d_m = to_keras_tensor(0. * loss + d_m)

    return loss, d_a, triuAAt, d_m
