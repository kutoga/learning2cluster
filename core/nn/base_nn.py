from math import ceil

import matplotlib.pyplot as plt
import pylab

from core.nn.helper import save_weights, load_weights, save_history, load_history, save_optimizer_state, \
    load_optimizer_state, sliding_window_average
from core.nn.history import History
from core.event import Event

class BaseNN:
    def __init__(self, name="NN_[CLASS]", debug_mode=False):
        self._name = name
        self._formatted_name = self._generate_formatted_name()
        self._debug_mode = None
        self._shared_layers = {}

        # Registered models? The weights of these models are automatically saved and loaded if the save and load methods
        # are used. The models may be Keras models or also "BaseNN" instances.
        self._registered_models = {}

        # All registered plots
        self._registered_plots = {}

        # Plot settings: If the given plot_sliding_window_average method is used, how large in percent/100 of all given
        # values should the averaging range be?
        self._plot_sliding_window_range_percentage = 0.05  # 0.05 => 5 percent

        # Training histories: The key is a keras model and the value a history object
        self._histories = {}

        # Some events
        self.event_store_weights_before = Event()
        self.event_store_weights_after = Event()
        self.event_load_weights_before = Event()
        self.event_load_weights_after = Event()
        self.event_plot_created = Event()
        self.event_debug_mode_changed = Event()
        self.event_debug_mode_on = Event()
        self.event_debug_mode_off = Event()

        self.debug_mode = debug_mode

    @property
    def debug_mode(self):
        return self._debug_mode

    @debug_mode.setter
    def debug_mode(self, debug_mode):
        if type(debug_mode) != bool:
            raise ValueError()
        if self._debug_mode == debug_mode:
            return

        self._debug_mode = debug_mode
        if self._debug_mode:
            self.event_debug_mode_on.fire()
        else:
            self.event_debug_mode_off.fire()
        self.event_debug_mode_changed.fire(self._debug_mode)

    @property
    def plot_sliding_window_range_percentage(self):
        return self._plot_sliding_window_range_percentage

    def _get_history(self, model):
        if model not in self._histories:
            self._histories[model] = History()
        return self._histories[model]

    def _register_model(self, model, model_name=None):
        if model_name is None:

            # No name is given, so we have to generate one.
            i = 0
            while True:
                model_name = self._get_name('model') + str(i)
                if model_name not in self._registered_models:
                    break
                i += 1

        self._registered_models[model_name] = model
        if not isinstance(model, BaseNN):
            print("Registered Keras model '{}' with '{}' parameters.".format(model_name, model.count_params()))

    def _clear_registered_models(self):
        self._registered_models.clear()

    def _register_plot(self, model_name, f_plot, figure=None, f_plot_if=None):
        """

        :param model_name:
        :param f_plot: A function with two parameters: the model-history and the matplotlib plot object
        :param f_plot_if: Only create this specific plot if a given condition is ok. If f_plot_if is None, no condition has to be satisfied.
        :return:
        """
        if figure not in self._registered_plots.keys():
            self._registered_plots[figure] = []
        self._registered_plots[figure].append({
            'model_name': model_name,
            'f_plot': f_plot,
            'f_plot_if': f_plot_if
        })

    def plot_sliding_window_average(self, values):

        # Add 0.1 to the length of all values to avoid some "0" problems
        return sliding_window_average(values, int(ceil((len(values) + 0.1) * self._plot_sliding_window_range_percentage)))

    def _clear_registered_plots(self, figure=None, all_figures=True):
        if all_figures and (figure is not None):
            self._registered_plots.clear()
        elif figure in self._registered_plots.keys():
            del self._registered_plots[figure]

    def _register_plots(self):
        """
        This method may be overwritten, but the super method always should be called. It should be called after all
        models are initialized.
        :return:
        """
        self._clear_registered_plots()

    def _generate_formatted_name(self):
        name = self._name
        name = name.replace("[CLASS]", type(self).__name__)
        return name

    def _get_name(self, name):
        return self._formatted_name + "_" + name

    def _s_layer(self, base_name, builder, format_name=True):
        """
        This function is a shortcut for self.__try_get_shared_layer(self.__get_name(base_name), builder)
        :param base_name:
        :param builder:
        :return:
        """
        if format_name:
            name = self._get_name(base_name)
        else:
            name = base_name
        return self._try_get_shared_layer(name, builder)

    def _try_get_shared_layer(self, name, builder):
        if name not in self._shared_layers:
            self._shared_layers[name] = builder(name)
        return self._shared_layers[name]

    def share_layers_between_networks(self, nn):
        """
        Share layers between different neural network objects. This operation must be done before any local shared
        layer is added (usually this is done when the network is initialized).

        :param nn: The "source" neural network.
        :return: Did it work (True|False)?
        """
        if len(self._shared_layers) > 0:
            return False
        self._shared_layers = nn._shared_layers

    def get_shared_layer(self, name):
        return self._shared_layers[name]

    def save_weights(self, base_filename, include_history=True, include_optimizer_state=True):
        base_filename += '_' + self._get_name('')

        self.event_store_weights_before.fire(base_filename, include_history)

        for model_name in self._registered_models:
            model = self._registered_models[model_name]
            current_base_filename = base_filename + model_name

            # There are two cases: The model is a Keras model, or it is a BaseNN model.
            if isinstance(model, BaseNN):
                model.save_weights(current_base_filename, include_history=include_history)
            else:
                save_weights(model, current_base_filename)
                if include_optimizer_state and hasattr(model, 'optimizer'):
                    save_optimizer_state(model, current_base_filename)
                if include_history:
                    save_history(self._get_history(model), current_base_filename)

        self.event_store_weights_after.fire(base_filename, include_history)

    def load_weights(self, base_filename, include_history=True, include_optimizer_state=True):
        base_filename += '_' + self._get_name('')

        self.event_load_weights_before.fire(base_filename, include_history)

        for model_name in self._registered_models.keys():
            model = self._registered_models[model_name]
            current_base_filename = base_filename + model_name

            # There are two cases: The model is a Keras model, or it is a BaseNN model.
            if isinstance(model, BaseNN):
                model.load_weights(current_base_filename, include_history=include_history)
            else:
                try:
                    load_weights(model, current_base_filename)
                    if include_optimizer_state and hasattr(model, 'optimizer'):
                        load_optimizer_state(model, current_base_filename)
                    if include_history:
                        self._get_history(model).load_keras_history(load_history(current_base_filename))
                except:
                    print('Could not load weights / history / optimizer state...')

        self.event_load_weights_after.fire(base_filename, include_history)

    def save_plots(self, output_base_filename=None):
        for figure_name in self._registered_plots.keys():
            registered_plots = self._registered_plots[figure_name]

            # Get all plots that have to be printed. We have to do this before the plotting itself, because the "subplot"
            # function already requires the complete count of plots. We also make the figure size dependend on the count
            # of subplots.
            plot_funcs = list(filter(
                lambda plot: plot['f_plot_if'] is None or
                             plot['f_plot_if'](self._histories[
                                self._registered_models[plot['model_name']]
                            ]),
                registered_plots
            ))
            plot_count = len(plot_funcs)

            plt.figure(1, (12, 2 * plot_count))
            fig = pylab.gcf()
            if output_base_filename is not None:
                if figure_name is not None:
                    output_filename = '{}_{}.png'.format(output_base_filename, figure_name)
                else:
                    output_filename = '{}.png'.format(output_base_filename)
                fig.canvas.set_window_title(output_filename)
                plt.title(output_filename)

            for i in range(len(plot_funcs)):
                plt.subplot(plot_count, 1, i + 1)
                plot = plot_funcs[i]
                model = self._registered_models[plot['model_name']]
                history = self._histories[model]
                plot['f_plot'](history, plt)

            self.event_plot_created.fire(plt)

            if output_filename is not None:
                plt.savefig(output_filename)
                plt.clf()
                plt.close()
            else:
                plt.show(block=True)
