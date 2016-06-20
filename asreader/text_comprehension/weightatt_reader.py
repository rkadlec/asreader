import theano.tensor as tt
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.initialization import Uniform
#from fuel.transformers import Batch, Padding, Cast, FilterSources

from custombricks.softmax_mask_bricks import SoftmaxWithMask

"""
Model for CBT task https://research.facebook.com/researchers/1543934539189348#cbt
Baseline results are reported in http://arxiv.org/abs/1511.02301
"""


from theano import tensor
import theano
from learning_experiment_base import str2bool
from blocks.bricks.lookup import LookupTable
from text_comprehension_base import TextComprehensionBase

def decorate(theano_var, name, level=1):
    if level > 3:
        return tt.printing.Print(name,("__str__","shape"))(theano_var)
    else:
        return theano_var




class TextComprehensionWeightedAtt(TextComprehensionBase):
    def __init__(self, **kwargs):

        super(TextComprehensionWeightedAtt, self).__init__(**kwargs)

    def add_command_line_args(self, parser):
        """
        Handler for adding custom command line arguments specific to a classifier.
        :param parser:
        :return:
        """

        super(TextComprehensionWeightedAtt, self).add_command_line_args(parser)

        parser.add_argument('-rsd', '--recurrent_stack_depth', type=int, default=2,
                            help='number of rnn layers in both encoder and decoder (meta param suggested value: {1;2;3;4})')

        parser.add_argument('-qice', '--query_inited_context_encoder', type=str2bool, default=False,
                            help='when true uses the text of the question to initialize context encoder, so the network knows the question and it can find only the relevant answers and not answers to all possible questions (meta param suggested value: {False;True})')

    def create_model(self, symbols_num = 500):

        hidden_states = self.args.encoder_hidden_dims
        embedding_dims = self.args.source_embeddings_dim

        # dimensions of sequence embeddings that are created bz bidir net, so the dimensionality is two times dim of a single net
        thought_dim = hidden_states * 2

        #query_dims = self.args.recurrent_stack_depth * self.args.encoder_hidden_dims

        # batch X input symbols
        context = tt.lmatrix('context')
        context_mask = tt.matrix('context_mask')
        context_mask = decorate(context_mask, "context_mask",level=1)
        # batch X output symbols
        x = tt.lmatrix('question')
        x_mask = tt.matrix('question_mask')
        # answer ix for each example in the batch
        y = tt.lmatrix('answer')


        # candidate answer words for each example, batch X candidate words (10 per each example)
        candidates_bi = tt.lmatrix("candidates")
        candidates_bi_mask = tt.matrix("candidates_mask")


        # TODO y can contain long sequences, here we use just the first symbol of each answer (that is possibly longer)
        # this have to be adjusted when response can be a sequence and not only a symbol
        y = decorate(y, "output")
        y = y[:,0]


        ###################
        # create model parts
        ###################

        lookup = LookupTable(symbols_num, embedding_dims, weights_init=Uniform(width=0.2))

        context_encoder = self.create_bidi_encoder("context_encoder", embedding_dims, hidden_states)

        question_encoder = self.create_bidi_encoder("question_encoder", embedding_dims, hidden_states)


        # inits
        lookup.initialize()
        #rnn.initialize()


        ###################
        # wire the model together
        ###################

        context = decorate(context, "CONTEXT",1)

        context_embedding_tbf = lookup.apply(context.T)
        #memory_encoded_btf = rnn.apply(context_embedding_tbf[:,0,:])[1]  # use cells
        memory_encoded_btf = context_encoder.apply(context_embedding_tbf.T,context_mask).dimshuffle(1,0,2)

        memory_encoded_btf.name = "memory_encoded_btf"
        memory_encoded_btf = decorate(memory_encoded_btf,"MEM ENC")

        # batch X features
        x = decorate(x,"X")
        x_embedded_btf = lookup.apply(x.T)
        x_embedded_btf = decorate(x_embedded_btf,"QUESTION EMB")
        x_encoded_btf = question_encoder.apply(x_embedded_btf.T, x_mask).dimshuffle(1,0,2)
        x_last = x_encoded_btf[-1]
        # extract forward rnn that is the first in bidir encoder
        x_encoded_btf = decorate(x_encoded_btf,"QUESTION ENC")

        x_forward_encoded_bf = x_encoded_btf[:,-1,0:hidden_states]
        x_backward_encoded_bf = x_encoded_btf[:,0,hidden_states:hidden_states*2]

        query_representation_bf = tt.concatenate([x_forward_encoded_bf,x_backward_encoded_bf],axis=1)

        # bidirectional representation of question is used as the search key
        search_key = query_representation_bf
        #search_key = x_last

        #search_key = W_um.apply(x_encoded)
        search_key = decorate(search_key,"SEARCH KEY")

        mem_attention_pre = tt.batched_dot(search_key, memory_encoded_btf.dimshuffle(0,2,1))
        mem_attention_pre = decorate(mem_attention_pre,"ATT presoftmax")

        # use masking on attention, this might be unnecessary but we do it just to be sure
        mem_attention_pre_masked_bt = tt.mul(mem_attention_pre,context_mask)
        mem_attention_pre_masked_bt = decorate(mem_attention_pre_masked_bt,"ATT presoftmax masked")

        #mem_attention_bt = Softmax(name="memory_query_softmax").apply(mem_attention_pre_masked_bt)
        mem_attention_bt = SoftmaxWithMask(name="memory_query_softmax").apply(mem_attention_pre_masked_bt,context_mask)

        mem_attention_bt = decorate(mem_attention_bt,"ATT",level=2)

        # compute weighted attention over original word vectors
        att_weighted_responses_bf = theano.tensor.batched_dot(mem_attention_bt, context_embedding_tbf.dimshuffle(1,0,2))

        #use mask to remove the probability mass from the unmasked candidates
        #word_probs_bi = word_probs_bi * candidates_bi_mask

        # compare desired response to all candidate responses
        # select relevant candidate answer words
        candidates_embeddings_bfi = lookup.apply(candidates_bi).dimshuffle(0,2,1)

        # convert it to output symbol probabilities
        y_hat_presoft = tt.batched_dot(att_weighted_responses_bf, candidates_embeddings_bfi)
        y_hat = SoftmaxWithMask(name="output_softmax").apply(y_hat_presoft,candidates_bi_mask)

        y_hat.name = "y_hat"
        y_hat = decorate(y_hat,"y_hat",level=2)

        # the correct answer is always the first among the candidates, so we can use zeros as index of ground truth
        y = y.zeros_like()

        # cost associated with prediction error
        cost_prediction = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
        cost_prediction.name = "cost_prediction"

        cost = cost_prediction

        attention_cost_weight = None

        cost_attention = None

        cost.name = "cost"


        predicted_response_index = tt.argmax(y_hat,axis=1)
        accuracy = tt.eq(y,predicted_response_index).mean()
        accuracy.name = "accuracy"

        return cost, accuracy, mem_attention_bt, y_hat, attention_cost_weight, cost_prediction, cost_attention, context, candidates_bi, candidates_bi_mask, y, context_mask, x, x_mask

exp = TextComprehensionWeightedAtt()
exp.execute()
