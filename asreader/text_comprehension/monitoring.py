from blocks.extensions.monitoring import MonitoringExtension
from blocks.monitoring.evaluators import MonitoredQuantityBuffer
from blocks.utils import dict_subset
from blocks.monitoring.aggregation import (MonitoredQuantity)
from blocks.utils import reraise_as
from blocks.extensions import SimpleExtension
from blocks.monitoring.evaluators import AggregationBuffer
from collections import OrderedDict
import logging
import numpy
import os
import pickle
import theano
import visualisation


__author__ = 'kadlec'


"""
This is mostly cut and paste code from DataStreamMonitoring class included in Blocks.
This would deserve to be rewritten from scratch.
"""

logger = logging.getLogger(__name__)


class MemoryDataStreamMonitoring(SimpleExtension, MonitoringExtension):
    """Monitors Theano variables and monitored-quantities on a data stream.

    By default monitoring is done before the first and after every epoch.

    Parameters
    ----------
    variables : list of :class:`~tensor.TensorVariable` and
        :class:`MonitoredQuantity`
        The variables to monitor. The variable names are used as record
        names in the logs.
    updates : list of tuples or :class:`~collections.OrderedDict` or None
        :class:`~tensor.TensorSharedVariable` updates to be performed
        during evaluation. This parameter is only for Theano variables.
        Be careful not to update any model parameters as this is not
        intended to alter your model in any meaningful way. A typical
        use case of this option arises when the theano function used
        for evaluation contains a call to :func:`~theano.scan` which
        might have returned shared variable updates.
    data_stream : instance of :class:`.DataStream`
        The data stream to monitor on. A data epoch is requested
        each time monitoring is done.

    """
    PREFIX_SEPARATOR = '_'

    def __init__(self, context_attention, context, y_hat, candidates, candidates_mask, y, context_mask, x, x_mask, data_stream=None, dictionary=None,
                 updates=None, output_dir=".", output_file="visual",
                 add_iteration_suffix=True, y_hat_out_file="", print_html=True, output_results=True,
                 **kwargs):

        kwargs.setdefault("before_first_epoch", True)

        self.context_attention = context_attention
        self.context = context
        self.y_hat = y_hat
        self.y = y
        self.candidates = candidates
        self.candidates_mask = candidates_mask
        self.dictionary = dictionary
        self.x = x
        self.output_dir = output_dir
        self.output_file = output_file
        self.add_iteration_suffix = add_iteration_suffix
        self.y_hat_out_file = y_hat_out_file
        self.print_html = print_html
        self.output_results = output_results

        self.code2token = {code: token for token, code in dictionary.iteritems()}

        super(MemoryDataStreamMonitoring, self).__init__(**kwargs)
        self._evaluator = MemoryDatasetEvaluator([context_attention, context,y_hat,y,candidates, candidates_mask, context_mask,x_mask,x], updates)
        self.data_stream = data_stream

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("Monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream)

        # create output file path
        epochs = self.main_loop.status['epochs_done']
        iters = self.main_loop.status['iterations_done']

        if self.add_iteration_suffix:
            filename = self.output_file+".e{}i{}.html".format(epochs,iters)
            predictions_filename = self.output_file+".e{}i{}.prediction".format(epochs,iters)
        else:
            filename = self.output_file+".html"
            predictions_filename = self.output_file+".prediction".format(epochs,iters)

        output_file_path = os.path.join(self.output_dir,filename)
        predictions_file_path = os.path.join(self.output_dir,predictions_filename)



        def ints_to_words(ints_list):
            """
            Translates a sequence of integers to a sequence of words using current code2token mapping.
            :param ints_list:
            :return:
            """
            return [self.code2token[j] for j in ints_list]

        batches_num = len(value_dict[self.context.name])
        correct_num = 0
        examples_num = 0

        data = []

        for j in xrange(batches_num):

            def get_value(theano_var,i):
                return value_dict[theano_var.name][j][i]

            def get_strs(theano_var,i):
                ints_val = get_value(theano_var,i)
                return ints_to_words(ints_val)

            def get_str_masked(theano_string_var, i):
                word_ixs = get_value(theano_string_var, i)
                mask = value_dict[theano_string_var.name+"_mask"][j][i]
                # create string
                # compute number of real words
                true_length = numpy.sum(mask)
                return ints_to_words(word_ixs[:true_length])


            for i in xrange(len(value_dict[self.context.name][j])):

                context_strs = get_str_masked(self.context,i)
                attention_ctx_i = get_value(self.context_attention,i)
                candidates_str = get_str_masked(self.candidates,i)
                y_hat_i = get_value(self.y_hat,i)
                y = get_value(self.y,i)
                question = get_str_masked(self.x,i)


                # add human readable data to a list
                data.append((" ".join(question), context_strs, attention_ctx_i, candidates_str, y_hat_i, 0))

                # test if prediction is correct
                correct_ix = numpy.argmax(y_hat_i)
                examples_num += 1
                if correct_ix == 0:
                    correct_num += 1


        fraction_correct = correct_num / (1.0 * examples_num)
        print "Accuracy {}".format(fraction_correct)

        if self.output_results:
            with open(predictions_file_path, 'w') as out_file:
                for output in data:
                    y_hat = output[4]
                    line_str = ", ".join(map(lambda x: str(x), y_hat)) + "\n"
                    out_file.write(line_str)

        if self.print_html:
            visualisation.make_html_file(data, output_file_path)
            logger.info("Visualization of attention stored in {}".format(output_file_path))

        # Save the predictions y_hat from the data list
        if self.y_hat_out_file:
            y_hat_out_fh = open(self.y_hat_out_file, 'w')
            pickle.dump(list([row[4] for row in data]), y_hat_out_fh)
            y_hat_out_fh.close()

        # store accuracy computed internally per example, not average over batch accuracies as is the default in Blocks
        report = {"accuracy": fraction_correct}

        self.add_records(self.main_loop.log, report.items())
        logger.info("Monitoring on auxiliary data finished")



class MemoryDatasetEvaluator(object):
    """A DatasetEvaluator evaluates many Theano variables or other quantities.

    The DatasetEvaluator provides a do-it-all method, :meth:`evaluate`,
    which computes values of ``variables`` on a dataset.

    Alternatively, methods :meth:`initialize_aggregators`,
    :meth:`process_batch`, :meth:`get_aggregated_values` can be used with a
    custom loop over data.

    The values computed on subsets of the given dataset are aggregated
    using the :class:`AggregationScheme`s provided in the
    `aggregation_scheme` tags. If no tag is given, the value is **averaged
    over minibatches**. However, care is taken to ensure that variables
    which do not depend on data are not unnecessarily recomputed.

    Parameters
    ----------
    variables : list of :class:`~tensor.TensorVariable` and
        :class:`MonitoredQuantity`
        The variable names are used as record names in the logs. Hence, all
        the names must be different.

        Each variable can be tagged with an :class:`AggregationScheme` that
        specifies how the value can be computed for a data set by
        aggregating minibatches.
    updates : list of tuples or :class:`~collections.OrderedDict` or None
        :class:`~tensor.TensorSharedVariable` updates to be performed
        during evaluation. This parameter is only for Theano variables.
        Be careful not to update any model parameters as this is not
        intended to alter your model in any meaningfullway. A typical
        use case of this option arises when the theano function used
        for evaluation contains a call to:function:`~theano.scan` which
        might have returned shared variable updates.

    """
    def __init__(self, variables, updates=None):

        theano_variables = []
        monitored_quantities = []
        for variable in variables:
            if isinstance(variable, MonitoredQuantity):
                monitored_quantities.append(variable)
            else:
                theano_variables.append(variable)
        self.theano_variables = theano_variables
        self.monitored_quantities = monitored_quantities
        variable_names = [v.name for v in variables]
        if len(set(variable_names)) < len(variables):
            raise ValueError("variables should have different names")
        self.theano_buffer = AggregationBuffer(theano_variables)
        self.monitored_quantities_buffer = MonitoredQuantityBuffer(
            monitored_quantities)
        self.updates = updates
        self._compile()

    def _compile(self):
        """Compiles Theano functions.

        .. todo::

            The current compilation method does not account for updates
            attached to `ComputationGraph` elements. Compiling should
            be out-sourced to `ComputationGraph` to deal with it.

        """

        #inputs = self.monitored_quantities_buffer.inputs
        #outputs = self.theano_variables

        inputs = []
        outputs = []


        updates = None

        if self.theano_buffer.accumulation_updates:
            updates = OrderedDict()
            updates.update(self.theano_buffer.accumulation_updates)
            inputs += self.theano_buffer.inputs
        if self.updates:
            # Handle the case in which we dont have any theano variables
            # to evaluate but we do have MonitoredQuantity
            # that may require an update of their own
            if updates is None:
                updates = self.updates
            else:
                updates.update(self.updates)
        inputs += self.monitored_quantities_buffer.inputs
        outputs = self.theano_variables

        if inputs != []:
            self.unique_inputs = list(set(inputs))
            self._accumulate_fun = theano.function(self.unique_inputs,
                                                   outputs)

        else:
            self._accumulate_fun = None

    def initialize_aggregators(self):
        self.theano_buffer.initialize_aggregators()
        self.monitored_quantities_buffer.initialize()

    def process_batch(self, batch):
        try:
            input_names = [v.name for v in self.unique_inputs]
            batch = dict_subset(batch, input_names)
        except KeyError:
            reraise_as(
                "Not all data sources required for monitoring were"
                " provided. The list of required data sources:"
                " {}.".format(input_names))
        if self._accumulate_fun is not None:
            numerical_values = self._accumulate_fun(**batch)
            for value, var in zip(numerical_values,self.theano_variables):
                self.data[var.name].append(value)

    def get_aggregated_values(self):
        values = self.theano_buffer.get_aggregated_values()
        values.update(
            self.monitored_quantities_buffer.get_aggregated_values())
        return values

    def evaluate(self, data_stream):
        """Compute the variables over a data stream.

        Parameters
        ----------
        data_stream : instance of :class:`.DataStream`
            The data stream. Only the first epoch of data is used.

        Returns
        -------
        A mapping from record names to the values computed on the provided
        dataset.

        """
        self.data = {var.name: [] for var in self.theano_variables}
        if self._accumulate_fun is not None:
            for batch in data_stream.get_epoch_iterator(as_dict=True):
                self.process_batch(batch)
        else:
            logger.debug(
                'Only data independent variables were given,'
                'will not iterate the over data!')

        return self.data
