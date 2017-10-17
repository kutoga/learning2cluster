from core.nn.base_nn import BaseNN


class EmbeddingNN(BaseNN):
    def __init__(self):
        super().__init__()
        self._model = None
        self._regularizer = None

    @property
    def model(self):
        return self._model

    @property
    def regularizer(self):
        return self._regularizer

    @regularizer.setter
    def regularizer(self, regularizer):
        self._regularizer = regularizer

    def _build_model(self, input_shape):
        pass

    def build(self, input_shape):
        self._model = self._build_model(input_shape)
        # self._model.summary()

        self._register_model(self._model)
