import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Activation, BatchNormalization, InputSpec, Layer, Lambda, RepeatVector, Reshape, \
    multiply, add, Flatten
from keras.regularizers import l2
import keras.regularizers
import keras.initializers
import keras.backend as K

class DeltaF(Layer):
    def __init__(self, alpha_regularizer=None, default_layer_count=1.,
                 **kwargs):
        super(DeltaF, self).__init__(**kwargs)
        self.supports_masking = True
        if default_layer_count < K.epsilon():
            print("The default layer count should never be smaller or equal to 0, otherwise it will never change (alsp depending on the alpha regularizer). It is recommended to use at least 0.5.")
        self.default_layer_count = default_layer_count
        self.alpha_initializer = keras.initializers.Constant(default_layer_count)
        self.alpha_regularizer = keras.regularizers.get(alpha_regularizer)

    def build(self, input_shape):
        param_shape = [1]
        self.alpha = self.add_weight(shape=param_shape,
                                     name='alpha',
                                     regularizer=self.alpha_regularizer,
                                     initializer=self.alpha_initializer)
        # Set input spec
        self.input_spec = InputSpec(ndim=len(input_shape))
        self.built = True

    def call(self, inputs, mask=None):
        return K.relu(1 - K.exp(inputs - self.alpha))

    def get_config(self):
        config = {
            'alpha_initializer': keras.initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': keras.regularizers.serialize(self.alpha_regularizer),
            'default_layer_count': self.default_layer_count
        }
        base_config = super(DeltaF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DynamicLayer:
    def __init__(self, name, f_layer, h_step=[lambda reg: Activation('relu')], w_init=1.0,
                 w_regularizer=(l2, 'auto'), f_regularizer=(l2, 1e-8), h_regularizer=None, res_net_like=True,):
        # w_regularizer[1] = 'auto' oder 'f_regularizer' = auto =>
        if not isinstance(f_layer, list):
            f_layer = [f_layer]
        if not isinstance(h_step, list):
            h_step = [h_step]

        self.name = name
        self.f_layer = f_layer
        self.h_step = h_step
        self.w_init = w_init
        self.w_regularizer = w_regularizer
        self.f_regularizer = f_regularizer
        self.h_regularizer = h_regularizer
        self.parameter_count_per_f = None
        self._current_layers = []
        self._deltaF = None
        self._res_net_like = res_net_like
        self._regularization_weight = 1.0

    def _get_regularizer_c(self):
        # w_regularizer = parameter_count_per_f * f_regularizer * c
        c = 5e-2 / (104 * 1e-5)
        return c

    def _get_regularizer(self, current, other, c_f):
        if current is None:
            return None
        if current[1] == 'auto':
            current = (current[0], other[1] * c_f)
        current = (current[0], current[1] * self._regularization_weight)
        return current[0](current[1])

    def _get_w_regularization(self):
        return self._get_regularizer(self.w_regularizer, self.f_regularizer, c_f=self._get_regularizer_c() * self.parameter_count_per_f)

    def _get_f_regularizer(self):
        return self._get_regularizer(self.f_regularizer, self.w_regularizer, c_f=1/(self._get_regularizer_c() * self.parameter_count_per_f))

    def reweight_regularization(self, regularization_weight):
        self._regularization_weight = regularization_weight

    def get_w(self):
        weights = self._deltaF.get_weights()
        if len(weights) == 0:
            return self.w_init
        return weights[0][0]

    def get_current_parameter_count(self):
        return max(0, int(np.ceil(self.get_w()))) * self.parameter_count_per_f

    def build_network(self, nw):
        layer_count = int(np.ceil(self.get_w()))
        dim = list(map(lambda x: int(str(x)), nw._keras_shape[1:]))
        dim_n = np.prod(dim)

        # Remove layers if we have too many of them
        self._current_layers = self._current_layers[:layer_count]

        # Add new layers if more layers are required
        for i in range(len(self._current_layers), layer_count):
            self._current_layers.append({
                'f_layer': self._get_layers('f', self.f_layer, self._get_f_regularizer()),
                'h_step': self._get_layers('f', self.h_step, self.h_regularizer)
            })

        # Define a helper function to get constant values. They must be included in the graph, therefore it is a bit hacky (probably there are nicer ways to do this)
        dummy_layer = nw
        if len(dim) > 1:
            dummy_layer = Flatten()(dummy_layer)
        dummy_layer = Dense(1, kernel_initializer='zeros', trainable=False)(nw)
        def constant(c):
            return Lambda(lambda s: s * 0 + c)(dummy_layer)

        # Build now the model; Call the input "x"
        x = nw
        for i in range(layer_count):

            # Execute the delta-function for the current layer (i)
            lf = self._deltaF(constant(i))

            # Resize it to the required dimension
            lf = RepeatVector(dim_n)(lf)
            lf = Reshape(dim)(lf)

            # Calculate the next layer
            l_next = x
            for layer in self._current_layers[i]['f_layer']:
                l_next = layer(l_next)

            # Weight the new calculation
            l_next = multiply([l_next, lf])

            # If required: Weight the old calculation
            if not self._res_net_like:
                lf_i = Lambda(lambda x: 1 - x)(lf)
                x = multiply([x, lf_i])

            # Sum up the two values:
            x = add([l_next, x])

            # Calculate h_step
            for layer in self._current_layers[i]['h_step']:
                x = layer(x)

        nw = x
        return nw

    def init(self, input_layer):
        layers = self._get_layers('f', self.f_layer)

        # Build the "subnetwork"
        nw = input_layer
        for layer in layers:
            nw = layer(nw)

        # Calculate the parameter count
        parameters = 0
        for layer in layers:
            parameters += sum(list(map(lambda w: np.prod(w.shape), layer.get_weights())))
        self.parameter_count_per_f = parameters

        # Create the DeltaF layer
        self._deltaF = DeltaF(self._get_w_regularization(), default_layer_count=self.w_init)

    def _get_layers(self, base_name, layer_builders, regularizer=None):
        return [layer_builders[i](regularizer) for i in range(len(layer_builders))]

class TDModel:
    def __init__(self):
        self._layers = []
        self._model = None
        self._dynamic_layers = []
        self._current_dynamic_config_key = None
        self._compile_kwargs = {}

    def append(self, layer):
        if isinstance(layer, list):
            for l in layer:
                self.append(l)
            return
        self._layers.append(layer)

    def __add__(self, layer):
        self.append(layer)
        return self

    def _get_dynamic_config_key(self):
        return '_'.join(map(
            lambda l: str(max(0, int(np.ceil(l.get_w())))),
            self._dynamic_layers
        ))

    def _rebuild_model_if_required(self):
        key = self._get_dynamic_config_key()
        if self._current_dynamic_config_key == key:
            return
        print("Rebuild model...")

        # A new key: This means we need to build a new model
        nw_input = self._layers[0]
        nw = nw_input
        for layer in self._layers[1:]:
            if isinstance(layer, DynamicLayer):
                nw = layer.build_network(nw)
            else:
                nw = layer(nw)

        # All layers are created. Compile now the model
        self._model = Model(nw_input, nw)
        self._model.compile(**self._compile_kwargs)

        # Define the current model key
        self._current_dynamic_config_key = key

        print("Model parameter count: {}".format(self._model.count_params()))

    def init(self, **compile_kwargs):
        assert len(self._layers) > 0
        assert isinstance(self._layers[0], Input((1,)).__class__)

        # Prepare the dynamic layers
        nw = self._layers[0]
        for layer in self._layers[1:]:
            if isinstance(layer, DynamicLayer):

                # We need to calculate the parameter count of the dynamic layer
                layer.init(nw)

                # "Register" the dynamic layer
                self._dynamic_layers.append(layer)

            else:
                nw = layer(nw)

        # Set a regularization factor for all dynamic layers
        avg_parameters_per_layer = np.mean(list(map(
            lambda l: l.parameter_count_per_f,
            self._dynamic_layers
        )))
        for layer in self._dynamic_layers:
            layer.reweight_regularization(layer.parameter_count_per_f / avg_parameters_per_layer)

        # Define the compile arguments
        self._compile_kwargs = compile_kwargs

        # Build the initial model
        self._rebuild_model_if_required()

    def get_depths(self):
        return {
            layer.name: layer.get_w() for layer in self._dynamic_layers
        }

    def print_depths(self):
        for layer in self._dynamic_layers:
            print("{}.w = {}".format(layer.name, layer.get_w()))

    def train_step(self, x, y, validation_data=None, **kwargs):
        assert self._model is not None
        self._model.fit(x, y, validation_data=validation_data, **kwargs)
        self.print_depths()
        self._rebuild_model_if_required()

# Create a simple XOR model
units = 8
n_inputs = 4
n_used_inputs = 2
assert n_used_inputs <= n_inputs
r_c = 1e-3
model = TDModel()
model += Input((n_inputs,))
model += Dense(units, trainable=False)
model += DynamicLayer(
    'd0',
    f_regularizer=(l2, r_c*1e-5), #l2(r_c * 1e-5),
    w_regularizer=(l2, 'auto'), #l2(r_c * 5e-2),
    f_layer=[
        lambda reg: Dense(units, kernel_regularizer=reg),
        lambda reg: BatchNormalization()
    ], h_step=[
        lambda reg: Activation('relu')
    ])
model += DynamicLayer(
    'd1',
    f_regularizer=(l2, r_c*1e-5), #l2(r_c * 1e-5),
    w_regularizer=(l2, 'auto'), #l2(r_c * 5e-2),
    f_layer=[
        lambda reg: Dense(units, kernel_regularizer=reg),
        lambda reg: BatchNormalization()
    ], h_step=[
        lambda reg: Activation('relu')
    ])
model += Dense(1, trainable=False, activation='sigmoid')
model.init(
    optimizer='adadelta',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# A data generator
def generate_xor_data(n):
    x = np.random.uniform(0, 1, (n, n_inputs))
    x[x>=.5] = 1.
    x[x<.5] = 0.
    y = np.sum(x[:, :n_used_inputs], axis=1) % 2
    y[y > 1] = 0.
    return x, y

n = 500
depths_hist = []
for i in range(10000000):
    print("Iteration {}".format(i))
    x, y = generate_xor_data(n)
    # y *= 0 # use a constant output (less layers shoudl be required to produce a network)
    model.train_step(x, y, batch_size=n)
    depths_hist.append(model.get_depths())
    print()

# TODO: Autoscaling vom l2-Loss für den w-Regularizer erstellen, basierend auf der Annahme, dass gilt w_l2=anzahl_p * p_l2 / c
# 5e-2 = 104 * 1e-5 / c
# => c = 104 * 1e-3 / 5 = 104 * 2e-4
# => c = anzahl(p) * 2e-4

# w = p * f * c
# 5e-2 = 104 * 1e-5 * c
# c = 0.05/0.00001/104
# c = 5e-2 / (104 * 1e-5)