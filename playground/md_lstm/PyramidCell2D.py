import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple




class PyramidRNNCell(object):
    """Abstract object representing an Convolutional RNN cell.
    """

    def __call__(self, inputs, state, dimension, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        """
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros
        """

        shape = self.in_shape
        num_features = self.num_features
        zeros = tf.zeros([batch_size, shape[0], num_features*2])
        return zeros

    def init_variables(self, dimensions, input_depth, scope):
        # initialize the tensors necessary for the Pyramid-lstm cell
        # For each dimension eight convolutional filters. four [filter_size * input_channels * size_state] for inputs
        # and four [filter_size * size_state * size_state] for hiddenlt statss

        # Addititionally four biases []

        with tf.variable_scope(scope, reuse= False):
            for dimension in dimensions:
                for gate in ["input", "forget", "update", "output"]:
                    tf.get_variable(gate + "_x" + str(dimension), [self.filter_size[0], input_depth, self.num_features])
                    tf.get_variable(gate + "_h" + str(dimension), [self.filter_size[0], self.num_features, self.num_features])

                for gate in ["input", "forget", "update", "output"]:
                    tf.get_variable(gate + "_bias" + str(dimension), [self.in_shape[abs(dimension)-1], self.num_features])
        return None


class BasicPyramidLSTMCell2D(PyramidRNNCell):
    # Basic 2D PyramidRNNCell


    def __init__(self, in_shape, filter_size, num_features, forget_bias=1.0,
                 state_is_tuple=False, activation=tf.nn.tanh):


        # Initialize basic Pyramid Cell
        # dimension of the input (length of one row/col of input)
        self.in_shape = in_shape
        # Size of the convolutional Filter used
        self.filter_size = filter_size
        # Size of the hidden State
        self.num_features = num_features
        # Forget Bias
        self._forget_bias = forget_bias
        # Tensorflow technicality
        self._state_is_tuple = state_is_tuple
        # Activation Function to be used within LSTM cell
        self._activation = activation


    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, dimension, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope):  # "basicPyramidCell"
            # Parameters of gates are concatenated into one multiply for efficiency.

            c, h = tf.split(axis=2, num_or_size_splits=2, value=state)

            i_d, f_d, c_tilde_d, o_d = conv1D([inputs, h], self.filter_size, self.num_features, self.in_shape, dimension, self._activation, scope=scope)

            new_c = tf.add(tf.multiply(c, f_d), tf.multiply(i_d, c_tilde_d))
            new_h = tf.multiply(self._activation(new_c), o_d)

            new_state = tf.concat(axis=2, values=[new_c, new_h])
            return new_h, new_state


def conv1D(args, filter_size, num_features, in_shape, dimension, activation, scope=None):

    # Calculate the total size of arguments on dimension 1.

    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 3:
            raise ValueError("Linear is expecting 3D arguments: %s" % str(shapes))
        if not shape[2]:
            raise ValueError("Linear expects shape[2] of arguments: %s" % str(shapes))

    dtype = [a.dtype for a in args][0]

    in_with = in_shape[abs(dimension)-1]

    # Now the variables.
    scope.reuse_variables()
    with tf.variable_scope(scope):

        [x, h] = args

        input_depth = x.get_shape().as_list()[2]

        intermediate = list()
        for gate in ["input", "forget", "update", "output"]:

            theta_x_d = tf.get_variable(gate + "_x" + str(dimension), [filter_size[0], input_depth, num_features], dtype=dtype)
            theta_h_d = tf.get_variable(gate + "_h" + str(dimension), [filter_size[0], num_features, num_features], dtype=dtype)
            bias = tf.get_variable(gate + "_bias" + str(dimension), [in_with, num_features], dtype=dtype)

            res1 = tf.nn.conv1d(x, theta_x_d, stride=1, padding="SAME")
            res2 = tf.nn.conv1d(h, theta_h_d, stride=1, padding="SAME")
            res = tf.add(res1, res2)
            intermediate.append(tf.add(res, bias))

        i_d = tf.nn.sigmoid(intermediate[0])
        f_d = tf.nn.sigmoid(intermediate[1])
        c_tilde_d = activation(intermediate[2])
        o_d = tf.nn.sigmoid(intermediate[3])

    return i_d, f_d, c_tilde_d, o_d
