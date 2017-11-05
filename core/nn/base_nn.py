from itertools import chain, repeat
from math import ceil

import matplotlib.pyplot as plt
import pylab

from core.nn.helper import save_weights, load_weights, save_history, load_history, save_optimizer_state, \
    load_optimizer_state, sliding_window_average
from core.nn.history import History
from core.event import Event

class BaseNN:
    def __init__(self, name="NN_[CLASS]", debug_mode=False, additional_debug_array_printer=None, prepend_base_name_to_layer_name=True):
        self._name = name
        self._formatted_name = self._generate_formatted_name()
        self._prepend_base_name_to_layer_name = prepend_base_name_to_layer_name
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
        self._plot_sliding_window_range_max_len = 500
        self._plot_sliding_window_max_values_for_calculation = 10000

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

        self._debug_mode = debug_mode
        self._additional_debug_array_printer = additional_debug_array_printer

    @property
    def prepend_base_name_to_layer_name(self):
        return self._prepend_base_name_to_layer_name

    @prepend_base_name_to_layer_name.setter
    def prepend_base_name_to_layer_name(self, prepend_base_name_to_layer_name):
        self._prepend_base_name_to_layer_name = prepend_base_name_to_layer_name

    @property
    def additional_debug_array_printer(self):
        return self._additional_debug_array_printer

    @additional_debug_array_printer.setter
    def additional_debug_array_printer(self, additional_debug_array_printer):
        self._additional_debug_array_printer = additional_debug_array_printer

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

    @property
    def plot_sliding_window_range_max_len(self):
        return self._plot_sliding_window_range_max_len

    @property
    def plot_sliding_window_max_values_for_calculation(self):
        return self._plot_sliding_window_max_values_for_calculation

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
        window_range = int(ceil((len(values) + 0.1) * self._plot_sliding_window_range_percentage))
        if self._plot_sliding_window_range_max_len is not None:
            window_range = min(self._plot_sliding_window_range_max_len, window_range)

        # BUGGY: If only every nh value is used, there are problems with the alidation data
        # # TODO: Implement a possibility to overjump every second / third / whatever value for the calculation of the average
        # if self._plot_sliding_window_max_values_for_calculation < len(values):
        #     assert self._plot_sliding_window_max_values_for_calculation > 0
        #     nth_value = int(ceil(len(values) / self._plot_sliding_window_max_values_for_calculation))
        #     input_values = values[::nth_value]
        # else:
        #     nth_value = None
        #     input_values = values
        #
        # result = sliding_window_average(input_values, window_range)
        #
        # if nth_value is not None:
        #     result = list(chain.from_iterable(repeat(x, nth_value) for x in result))[:len(values)]
        # return result

        return sliding_window_average(values, window_range)

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
        if self._prepend_base_name_to_layer_name:
            return self._formatted_name + "_" + name
        else:
            return name

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

        for model_name in sorted(self._registered_models.keys()):
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

        for model_name in sorted(self._registered_models.keys()):
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
                except Exception as e:
                    print('Could not load weights / history / optimizer state...')
                    # print('Could not load weights / history / optimizer state. Error: {}'.format(e.message.replace('\n', '; ')))

        self.event_load_weights_after.fire(base_filename, include_history)

    def save_plots(self, output_base_filename=None):
        print("Create all registered plots. Output base filename: {}".format(output_base_filename))
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
        print("All plots are done")
