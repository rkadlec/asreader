import codecs
import os
import re
from collections import OrderedDict

import nltk
from blocks.algorithms import Adam
from blocks.bricks import Initializable, Tanh, Linear
from blocks.bricks.base import application
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import Bidirectional, GatedRecurrent
from blocks.config import config
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.serialization import load
from fuel.schemes import ConstantScheme, ShuffledExampleScheme
from fuel.transformers import Batch, Padding, Mapping, SortMapping, Unpack
from picklable_itertools.extras import equizip
from theano import tensor
from toolz import merge

import cnn_ne_shuffle
from ariadne_config_argparse import get_current_metaparams_str
from customfuel.datasets import UnpickableIndexableDataset
from datasets.babi_dataset import bAbIDataset
from datasets.cbt_dataset import CBDataset
from datasets.cnn_dataset import CNNDataset
from learning_experiment_base import LearningExperimentBase, MultiOutputModel
from learning_experiment_base import str2bool
from monitoring import MemoryDataStreamMonitoring
from text_utilities import get_vocabulary, compute_token2code

"""
This is an implementation of the Attention Sum Reader (AS Reader) network as presented in
http://arxiv.org/abs/1603.01547
If you use this implementation in your work, please kindly cite the above paper.

Example usage: python as_reader.py --train training_data.txt --valid validation_data.txt --test test_data.txt -ehd 256 -sed 256
"""


config.recursion_limit = 100000


def _length(example):
    # return length of the context, this is usually the longest text field in the training example
    return len(example[0])

class BidirectionalFromDict(Bidirectional):
    """ Bidirectional RNN that takes inputs in form of a dict. """

    @application
    def apply(self, forward_dict, backward_dict):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(as_list=True, **forward_dict)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           **backward_dict)]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in equizip(forward, backward)]


class BidirectionalEncoder(Initializable):
    """ Bidirectional GRU encoder. """

    def __init__(self, embedding_dim, state_dim, **kwargs):
        super(BidirectionalEncoder, self).__init__(**kwargs)
        # Dimension of the word embeddings taken as input
        self.embedding_dim = embedding_dim
        # Hidden state dimension
        self.state_dim = state_dim

        # The bidir GRU
        self.bidir = BidirectionalFromDict(
            GatedRecurrent(activation=Tanh(), dim=state_dim))
        # Forks to administer the inputs of GRU gates
        self.fwd_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='fwd_fork')
        self.back_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='back_fork')

        self.children = [self.bidir,
                         self.fwd_fork, self.back_fork]

    def _push_allocation_config(self):
        self.fwd_fork.input_dim = self.embedding_dim
        self.fwd_fork.output_dims = [self.bidir.children[0].get_dim(name)
                                     for name in self.fwd_fork.output_names]
        self.back_fork.input_dim = self.embedding_dim
        self.back_fork.output_dims = [self.bidir.children[1].get_dim(name)
                                      for name in self.back_fork.output_names]


    @application(inputs=['source_sentence_tbf', 'source_sentence_mask_tb'],
                 outputs=['representation'])
    def apply(self, source_sentence_tbf, source_sentence_mask_tb=None):

        representation_tbf = self.bidir.apply(
            merge(self.fwd_fork.apply(source_sentence_tbf, as_dict=True),
                  {'mask': source_sentence_mask_tb}),
            merge(self.back_fork.apply(source_sentence_tbf, as_dict=True),
                  {'mask': source_sentence_mask_tb})
        )
        return representation_tbf


class TextComprehensionBase(LearningExperimentBase):
    """
    Base class that reads different text comprehension datasets, creates a model and starts training it.
    """
    def __init__(self, **kwargs):

        super(TextComprehensionBase, self).__init__(**kwargs)

    def add_command_line_args(self, parser):
        """
        Add custom command-line parameters
        :param parser:
        :return:
        """

        super(TextComprehensionBase, self).add_command_line_args(parser)

        parser.add_argument('-bv', '--batch_size_valid', type=int, default="100",
                            help='size of a mini-batch in test and validation')

        parser.add_argument('-skba', '--sort_k_batches_ahead', type=int, default=10,
                            help='number of batches that will be read in advance and sorted according to context length, this usually speeds up training')

        parser.add_argument('-dt', '--dataset_type', default='cbt', choices=['cbt','cnn', 'babi'],
                            help='type of the dataset to load')

        parser.add_argument('-ehd', '--encoder_hidden_dims', type=int, default=100,
                            help='number of recurrent hidden dimensions (meta param suggested value: {4;32;128;256})')

        parser.add_argument('-sed','--source_embeddings_dim', type=int, default=200,
                            help='dimensions of embeddings for source words (meta param suggested value: {100;300;600})')

        parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                            help='learning rate of the gradient update rule (meta param suggested value: {0.005;0.001;0.0005})')

        parser.add_argument('-qice', '--query_inited_context_encoder', type=str2bool, default=False,
                            help='when true uses the text of the question to initialize context encoder, so the network knows the question and it can look only for the relevant answer and not answers to all possible questions (meta param suggested value: {False;True})')

        parser.add_argument('--load_model', type=str, default=None,
                            help='model to be loaded')

        parser.add_argument('--output_dir', type=str, default='.',
                            help='output directory where, e.g., visualization will be stored')

        parser.add_argument('--own_eval', action='store_true', default=True,
                            help='whether to use our evaluation during training')

        parser.add_argument('--files_to_visualize', nargs="+", type=str,
                            help='list of files that will be visualized by the loaded model')

        parser.add_argument('--y_hat_out_file', type=str, default="",
                            help='dump file for y_hat generated by the loaded model')

        parser.add_argument('--no_html', dest='print_html', action='store_false',
                            help='disables html visualization printing')
        parser.set_defaults(print_html=True)

        parser.add_argument('--weighted_att', action='store_true',
                            help='Use weighted attention model instead of attention sum.')



    def create_bidi_encoder(self, name, embedding_dims, hidden_states):
        """
        Creates and initializes a bidirectional GRU encoder
        :param name:
        :param embedding_dims:
        :param hidden_states:
        :return:
        """


        encoder = BidirectionalEncoder(embedding_dims, hidden_states, name=name)

        # Set up the initialization methods
        weight_scale = 0.1
        encoder.weights_init = IsotropicGaussian(
            weight_scale)
        encoder.biases_init = Constant(0)

        encoder.push_initialization_config()
        encoder.bidir.prototype.weights_init = Orthogonal()
        encoder.initialize()

        return encoder


    def create_model(self, symbols_num = 500):
        """
        This method should be overridden by subclasses.
        :param symbols_num:
        :return:
        """
        None

    def execute(self):

        if self.args.output_dir:
            print "Output will be stored in {}".format(self.args.output_dir)
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)

        def get_stream(file, dictionary=None, add_dict=False, shuffle=False, batch_size=None, read_ahead=1):
            """
            Creates a stream with train/valid/test examples.
            :param file: path to a file with the dataset
            :param dictionary: string->int dict with vocabulary from other datasets. If None, the dictionary is built
                               from this dataset
            :param add_dict: if True, new words are added to the dictionary from this dataset
            :param shuffle: if True, the dataset examples are randomly shuffled
            :param batch_size
            :param read_ahead: Number of batches that shall be pre-fetched and ordered by context length to speed up computation

            """

            # Dataset type (CNN/CBT/bAbI)
            data_type = self.args.dataset_type

            if not batch_size:
                batch_size = self.args.batch_size

            # Pattern for text tokenization
            pattern = re.compile(" |\t|\|")

            if data_type == 'babi':
                prepro = lambda x : nltk.word_tokenize(x)
            else:
                prepro = lambda x : pattern.split(x)

            if add_dict:
                # add words to dictionary
                f = codecs.open(file, 'r', encoding="utf8")
                vocabulary = get_vocabulary(f, prepro)
                code2token = map(lambda x : x[0], vocabulary.most_common())

                new_word_count = 0
                for word in code2token:
                    if word not in dictionary:
                        dictionary[word] = len(dictionary)
                        new_word_count += 1

                print "Added {} new words from file {} to previous vocabulary.".format(new_word_count, file)

            if not dictionary:
                print "Computing new vocabulary for file {}.".format(file)
                # compute vocabulary
                f = codecs.open(file, 'r', encoding="utf8")
                vocabulary = get_vocabulary(f, prepro)
                code2token = map(lambda x : x[0], vocabulary.most_common())
                # Add special symbols (beginning/end of sentence, unknown token, end of question)
                code2token.extend(['<S>','</S>','<UNK>', '<QUESTION_END>'])
                dictionary = compute_token2code(code2token)


            # Select the data loader appropriate for the dataset

            common_params = {'level': 'word','bos_token': None,'eos_token':None,
                               'append_question': self.args.query_inited_context_encoder}

            if data_type == 'cnn':
                dataset = CNNDataset([file],dictionary,**common_params)
            elif data_type == 'cbt':
                dataset = CBDataset([file],dictionary,**common_params)
            elif data_type == 'babi':
                dataset = bAbIDataset([file],dictionary,**common_params)

            stream = dataset.get_example_stream()

            # Load all data into memory, this way we avoid reloading the data from disk in every epoch
            memory_data = [[] for _ in dataset.sources]
            for ex in stream.get_epoch_iterator():
                for source_example, data_list in zip(ex,memory_data):
                    data_list.append(source_example)

            data_dict = OrderedDict(zip(dataset.sources, memory_data))
            mem_dataset = UnpickableIndexableDataset(data_dict)
            if shuffle:
                # shuffle the data after each epoch of training
                mem_dataset.example_iteration_scheme = ShuffledExampleScheme(mem_dataset.num_examples)
            stream = mem_dataset.get_example_stream()


            # Build a batched version of stream to read k batches ahead
            stream = Batch(stream,
                           iteration_scheme=ConstantScheme(
                               batch_size*read_ahead))

            if read_ahead > 1:

                # Sort all samples in the read-ahead batch
                stream = Mapping(stream, SortMapping(_length))

                # Convert it into a stream again
                stream = Unpack(stream)

                # Construct batches from the stream with specified batch size
                stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))

            # Pad sequences that are short
            stream = Padding(stream, mask_sources=['context', 'question', 'candidates'])

            return stream, dictionary

        if self.args.load_model:
            # this code is executed only when the user loads already trained model

            with open(self.args.load_model, "rb") as model_file:
                print "Loading model {} ...".format(self.args.load_model)
                main_loop = load(model_file)  # load function from blocks.serialization
                model = main_loop.model

                cost, accuracy, mem_attention_bt, y_hat, context_bt, candidates_bi, candidates_bi_mask, y, context_mask_bt, question_bt, question_mask_bt = model.orig_outputs
                # TODO add words from new file
                for file in self.args.files_to_visualize:
                    # loop over multiple files that can be evaluated by the model

                    stream, vocab = get_stream(self.get_data_path(file), model.vocabulary)

                    # use only one extension that will visualize the datastream
                    extension = MemoryDataStreamMonitoring(mem_attention_bt, context_bt, y_hat, candidates_bi, candidates_bi_mask, y, context_mask_bt, question_bt, question_mask_bt,
                                                                   data_stream=stream, dictionary=vocab, prefix="valid",
                                                                   output_dir=self.args.output_dir, output_file=file, before_training=True, add_iteration_suffix=False,
                                                                   y_hat_out_file=self.args.y_hat_out_file, print_html=self.args.print_html)

                    extension.main_loop = main_loop

                    main_loop.extensions = [extension]

                    main_loop._run_extensions('before_training')

            exit(0)



        train_stream, vocab = get_stream(self.get_data_path_argparse("train"), shuffle=True,
                                         read_ahead=self.args.sort_k_batches_ahead)
        # reuse vocabulary from the training file for valid and test set
        valid_stream, _ = get_stream(self.get_data_path_argparse("valid"), vocab, add_dict=True,
                                     batch_size=self.args.batch_size_valid)

        # load multiple test streams
        test_streams = {}
        for test_file in self.args.test:
            test_file_path =  self.get_data_path(test_file)
            test_stream, _ = get_stream(test_file_path, vocab, add_dict=True,
                                    batch_size=self.args.batch_size_valid)
            test_streams[test_file] = test_stream

        #add mapping specific for cnn and dm datasets
        if self.args.dataset_type == 'cnn':
            # this ensures that tokens for named entities are randomly shuffled for each batch
            cnn_ne_shuffle.set_dictionary(vocab)
            train_stream = Mapping(train_stream, cnn_ne_shuffle.shuffle_ne)
            valid_stream = Mapping(valid_stream, cnn_ne_shuffle.shuffle_ne)
            test_streams = {k: Mapping(test_stream, cnn_ne_shuffle.shuffle_ne) for k, test_stream in test_streams.items()}


        model_outputs = self.create_model(len(vocab))
        cost, accuracy, mem_attention, y_hat, context, candidates, candidates_mask, y, context_mask, x, x_mask = model_outputs

        vars_to_monitor = [(cost,min),(accuracy,max)]

        extra_extensions = []



        if self.args.own_eval:
            valid_filename = os.path.basename(self.get_data_path_argparse('valid'))

            suffix = ""
            if self.args.append_metaparams:
               suffix = "."+get_current_metaparams_str(self.parser, self.args)

            # extension that outputs model's attention and predictions on valid set
            visualization_monitor_valid = MemoryDataStreamMonitoring(mem_attention, context, y_hat, candidates, candidates_mask, y, context_mask, x, x_mask,
                                                               data_stream=valid_stream, dictionary=vocab, prefix="valid",
                                                               output_dir=self.args.output_dir, output_file=valid_filename+suffix,
                                                               print_html=self.args.print_html,
                                                               **self.args.evaluate_every_n)

            # there can be multiple test streams
            test_monitors = []

            for test_filename, test_stream in test_streams.items():
                visualization_monitor_test = MemoryDataStreamMonitoring(mem_attention, context, y_hat, candidates, candidates_mask, y, context_mask, x, x_mask,
                                                               data_stream=test_stream, dictionary=vocab, prefix=test_filename,
                                                               output_dir=self.args.output_dir, output_file=test_filename+suffix,
                                                               print_html=self.args.print_html,
                                                               **self.args.evaluate_every_n)
                test_monitors.append(visualization_monitor_test)


            extra_extensions.extend([visualization_monitor_valid] + test_monitors)

        model = MultiOutputModel(model_outputs)
        # store vocabulary as part of the model, so we can easily reuse it after unpickling
        model.vocabulary = vocab


        self.train(cost,y_hat,train_stream,accuracy,valid_stream=valid_stream,
                   vars_to_monitor_on_train=vars_to_monitor,vars_to_monitor_on_valid=vars_to_monitor,
                   step_rule=Adam(learning_rate=self.args.learning_rate),
                   #additional_streams=[("test", test_stream)],
                   extra_extensions=extra_extensions,
                   model=model,
                   use_own_validation=True)
