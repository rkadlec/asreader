import sys

__author__ = 'rkadlec'

import argparse

from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.model import Model
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, AdaDelta, StepClipping, RemoveNotFinite, CompositeRule
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar, SimpleExtension
import os

from ariadne_config_argparse import create_ariadne_config_skeleton, get_current_metaparams_str
from custombricks.gradient_noise import GradientNoise

from custombricks.save_the_best import SaveTheBest

import logging

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


class FlushStreams(SimpleExtension):
    """Extension that flushes output stream during training. This should help when std out is redirected to a file.
    Parameters
    ----------
    """
    def __init__(self,  **kwargs):
        super(FlushStreams, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        sys.stdout.flush()



def training_progress(progress):
    """
    Translates a string in format {x}, {x}E or {x}B to a Blocks extension calling frequency
    :param progress:
    :return:
    """
    progress = progress.lower()
    if progress.endswith('e'):
        return {"every_n_epochs": int(progress[:-1])}
    elif progress.endswith('b'):
        return {"every_n_batches": int(progress[:-1])}
    else:
        return {"every_n_epochs": int(progress)}


def none_filter_fn(o):
    return o is not None

class MultiOutputModel(Model):
    """
    Model with multiple outputs. Only the first one is used as optimization objective.
    """
    def __init__(self, outputs):
        self.orig_outputs = outputs
        # filter out None elements
        filtered = filter(none_filter_fn,outputs)
        super(MultiOutputModel, self).__init__(filtered)


    def get_objective(self):
        """Return the output variable, if there is a single one.

        If there is only one output variable, it is a reasonable default
        setting to assume that it is the optimization objective.

        """
        return self.outputs[0]


class LearningExperimentBase(object):
    """
    Base class for training models based on Blocks. It wraps functionality provided by MainLoop and its extensions from Blocks.
    """

    def __init__(self, description="NA", epilog="NA"):
        self.args = self.get_command_line_args(description=description, epilog=epilog)
        self.logger = logging.getLogger(__name__)


    def add_command_line_args(self, parser):
        None

    def get_command_line_args(self, **kwargs):

        # parse commandline args

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs)

        parser.add_argument('--train', type=str, default=None,
                            help='train file')

        parser.add_argument('--valid', type=str, default=None,
                            help='valid file')

        parser.add_argument('--test', type=str, nargs='+', default=None,
                            help='test file')

        parser.add_argument('--dataset_root', type=str, default=None,
                            help='root directory with dataset files')

        parser.add_argument('-b', '--batch_size', type=int, default="32",
                            help='size of a mini-batch in gradient descend (meta param suggested value: {16;32;64} )')

        parser.add_argument('--save_every_n', type=training_progress, default="50",
                            help='frequency of saving the model during training')

        parser.add_argument('--evaluate_every_n', type=training_progress, default="1e",
                            help='frequency of evaluating performance on the train+validation set and possibly saving the best model')

        parser.add_argument('--create_ariadne_config', action='store_true',
                            help="creates configuration for Ariadne and Bivoj that will be stored in config/ subdir")

        parser.add_argument('--epochs_max', type=int, default=None,
                            help="maximum number of training epochs")

        parser.add_argument('-p', '--epochs_patience_valid', type=int, default=10,
                            help="maximum number of training epochs without improvement in validation set accuracy before end of the training")

        parser.add_argument('--patience_metric', choices=['cost','accuracy'], default='cost',
                            help="metric to monitor for early stopping")

        parser.add_argument('-gc', '--gradient_clip', type=float, default=10.0,
                            help='maximum magnitude of gradient, larger gradients will be clipped to this value (meta param suggested value: {1.0;10.0})')

        parser.add_argument('-gn', '--gradient_noise', type=float, default=0,
                            help='adds gaussian noise to the gradient and specifies parameter eta (suggested values: 0.01, 0.3, 1.0)')

        parser.add_argument('-gnd', '--gn_decay', type=float, default=0.1,
                            help='decay for the gradient decay (suggested values: 0.01, 0.1, 0.5)')

        parser.add_argument('--disable_progress_bar', action='store_true',
                            help="disables the progress bar that is shown during training, this might be handy when you want to store output of the training process in a file")

        parser.add_argument('--do_not_save', action='store_true',
                            help="do not save the model anytime, this option is usefull when one just wants to test convergence of the model")

        parser.add_argument('--save_path', dest='save_path', default="model.blocks.save",
                            help='file where the saved model will be stored')

        parser.add_argument('--append_metaparams', action='store_true',
                            help="append current metaparameters to saved model filename")

        parser.add_argument('-sob','--save_only_best', action='store_true',
                            help="save only the best model according to validation set")

        parser.add_argument('--save_parameters', type=str, default=None,
                            help="Save the model parameters at the end of training.")

        parser.add_argument('--verbose', action='store_true',
                            help="provide extra logging output")



        self.add_command_line_args(parser)

        args = parser.parse_args()

        if args.create_ariadne_config:
            # creates configuration file with all metaparameters
            create_ariadne_config_skeleton(parser)
            quit()

        self.parser = parser

        return args


    def get_data_path_argparse(self, arg_name):
        file_path = getattr(self.args,arg_name)
        return self.get_data_path(file_path)


    def get_data_path(self, file_path):
        if os.path.isabs(file_path):
            return file_path
        else:
            if self.args.dataset_root:
                return os.path.join(self.args.dataset_root, file_path)
            else:
                return file_path



    def train(self, cost, y_hat, train_stream, accuracy=None, prediction_cost=None, regularization_cost=None, params_to_optimize=None, valid_stream=None,
              extra_extensions=None, model=None, vars_to_monitor_on_train=None, vars_to_monitor_on_valid=None,
              step_rule=None, additional_streams=None, save_on_best=None, use_own_validation=False, objects_to_dump=None):
        """
        Generic method for training models. It extends functionality already provided by Blocks.
        :param cost: Theano var with cost function
        :param y_hat: Theano var with predictions from the model
        :param train_stream: Fuel stream with training data
        :param accuracy: Theano var with accuracy
        :param prediction_cost:
        :param regularization_cost:
        :param params_to_optimize:
        :param valid_stream: Fuel stream with validation data
        :param extra_extensions:
        :param model:
        :param vars_to_monitor_on_train:
        :param vars_to_monitor_on_valid:
        :param step_rule:
        :param additional_streams:
        :param save_on_best:
        :param use_own_validation:
        :param objects_to_dump:
        :return:
        """

        if not vars_to_monitor_on_valid:
            vars_to_monitor_on_valid = [(cost, min)]
            if accuracy:
                vars_to_monitor_on_valid.append((accuracy, max))


        if not save_on_best:
            # use default metrics for saving the best model
            save_on_best = [(cost, min)]
            if accuracy:
                save_on_best.append((accuracy, max))

        # setup the training algorithm #######################################
        # step_rule = Scale(learning_rate=0.01)
        #    step_rule = Adam()
        model_save_suffix = ""
        if self.args.append_metaparams:
            model_save_suffix = "."+get_current_metaparams_str(self.parser, self.args)

        # get a list of variables that will be monitored during training
        vars_to_monitor = [cost]
        if accuracy:
            vars_to_monitor.append(accuracy)
        if prediction_cost:
            vars_to_monitor.append(prediction_cost)
        if regularization_cost:
            vars_to_monitor.append(regularization_cost)


        theano_vars_to_monitor = [var for var, comparator in vars_to_monitor_on_valid]



        if not params_to_optimize:
            # use all parameters of the model for optimization
            cg = ComputationGraph(cost)
            params_to_optimize = cg.parameters

        self.print_parameters_info(params_to_optimize)

        if not model:
            if accuracy:
                model = MultiOutputModel([cost, accuracy, y_hat] + theano_vars_to_monitor)
            else:
                model = MultiOutputModel([cost, y_hat] + theano_vars_to_monitor)

        if not step_rule:
            step_rule = AdaDelta()  # learning_rate=0.02, momentum=0.9)

        step_rules=[StepClipping(self.args.gradient_clip), step_rule, RemoveNotFinite()]

        # optionally add gradient noise
        if self.args.gradient_noise:
            step_rules = [GradientNoise(self.args.gradient_noise, self.args.gn_decay)] + step_rules

        algorithm = GradientDescent(cost=cost,
                                    parameters=params_to_optimize,
                                    step_rule=CompositeRule(step_rules),
                                    on_unused_sources="warn")

        # this variable aggregates all extensions executed periodically during training
        extensions = []

        if self.args.epochs_max:
            # finis training after fixed number of epochs
            extensions.append(FinishAfter(after_n_epochs=self.args.epochs_max))

        # training data monitoring
        def create_training_data_monitoring():
            if "every_n_epochs" in self.args.evaluate_every_n:
                return TrainingDataMonitoring(vars_to_monitor, prefix='train', after_epoch=True)
            else:
                return TrainingDataMonitoring(vars_to_monitor, prefix='train', after_epoch=True, **self.args.evaluate_every_n)

        # add extensions that monitors progress of training on train set
        extensions.extend([
                      create_training_data_monitoring()])

        if not self.args.disable_progress_bar:
            extensions.append(ProgressBar())

        def add_data_stream_monitor(data_stream, prefix):
            if not use_own_validation:
                extensions.append(DataStreamMonitoring(variables=theano_vars_to_monitor,
                                                   data_stream=data_stream,
                                                   prefix=prefix,
                                                   before_epoch=False,
                                                   **self.args.evaluate_every_n
                                                    ))

        # additional streams that should be monitored
        if additional_streams:
            for stream_name, stream in additional_streams:
                add_data_stream_monitor(stream, stream_name)

        # extra extensions need to be called before Printing extension
        if extra_extensions:
            extensions.extend(extra_extensions)

        if valid_stream:
            # add validation set monitoring
            add_data_stream_monitor(valid_stream, 'valid')

            # add best val monitoring
            for var, comparator in vars_to_monitor_on_valid:
                extensions.append(TrackTheBest("valid_" + var.name, choose_best=comparator, **self.args.evaluate_every_n))

            if self.args.patience_metric == 'cost':
                patience_metric_name = cost.name
            elif self.args.patience_metric == 'accuracy':
                patience_metric_name = accuracy.name
            else:
                print "WARNING: Falling back to COST function for patience."
                patience_metric_name = cost.name

            extensions.append(
                              # "valid_cost_best_so_far" message will be entered to the main loop log by TrackTheBest extension
                              FinishIfNoImprovementAfter("valid_" + patience_metric_name + "_best_so_far", epochs=self.args.epochs_patience_valid)
                              )

            if not self.args.do_not_save:

                # use user provided metrics for saving
                valid_save_extensions = map(lambda metric_comparator: SaveTheBest("valid_" + metric_comparator[0].name,
                                                           self.args.save_path + ".best." + metric_comparator[0].name + model_save_suffix,
                                                           choose_best=metric_comparator[1], **self.args.evaluate_every_n),
                        save_on_best)
                extensions.extend(valid_save_extensions)



        extensions.extend([Timing(**self.args.evaluate_every_n),
                      Printing(after_epoch=False, **self.args.evaluate_every_n),
                      ])

        if not self.args.do_not_save or self.args.save_only_best:
            extensions.append(Checkpoint(self.args.save_path+model_save_suffix, **self.args.save_every_n
                                 ))


        extensions.append(FlushStreams(**self.args.evaluate_every_n))


        # main loop ##########################################################
        main_loop = MainLoop(data_stream=train_stream,
                             model=model,
                             algorithm=algorithm,
                             extensions=extensions
                             )
        sys.setrecursionlimit(1000000)
        main_loop.run()


    @staticmethod
    def print_parameters_info(params_list):
        print "Parameters: "
        for parameter in params_list:
            print "\t" + str(parameter.tag.annotations[0].name) + "." + str(parameter) + " " + str(parameter.container.data.shape) + ".size=" + str(
                parameter.container.data.size)
        print "Trained parameters count: " + str(sum([parameter.container.data.size for parameter in params_list]))

