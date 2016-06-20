import theano
import theano.tensor as tt
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks.lookup import LookupTable
from blocks.initialization import Uniform

from custombricks.softmax_mask_bricks import SoftmaxWithMask
from text_comprehension_base import TextComprehensionBase

"""
This is an implementation of the Attention Sum Reader (AS Reader) network as presented in
http://arxiv.org/abs/1603.01547
If you use this implementation in your work, please kindly cite the above paper.

Example usage: python as_reader.py --train training_data.txt --valid validation_data.txt --test test_data.txt -ehd 256 -sed 256
"""


class ASReader(TextComprehensionBase):
    def __init__(self, **kwargs):
        super(ASReader, self).__init__(**kwargs)


    def create_model(self, symbols_num = 500):

        # Hyperparameters

        # The dimension of the hidden state of the GRUs in each direction.
        hidden_states = self.args.encoder_hidden_dims
        # Dimension of the word-embedding space
        embedding_dims = self.args.source_embeddings_dim


        ###################
        # Declaration of the Theano variables that come from the data stream
        ###################

        # The context document.
        context_bt = tt.lmatrix('context')
        # Context document mask used to distinguish real symbols from the sequence and padding symbols that are at the end
        context_mask_bt = tt.matrix('context_mask')

        # The question
        question_bt = tt.lmatrix('question')
        question_mask_bt = tt.matrix('question_mask')

        # The correct answer
        y = tt.lmatrix('answer')
        y = y[:,0] # originally answers are in a 2d matrix, here we convert it to a vector

        # The candidates among which the answer is selected
        candidates_bi = tt.lmatrix("candidates")
        candidates_bi_mask = tt.matrix("candidates_mask")



        ###################
        # Network's components
        ###################

        # Lookup table with randomly initialized word embeddings
        lookup = LookupTable(symbols_num, embedding_dims, weights_init=Uniform(width=0.2))

        # bidirectional encoder that translates context
        context_encoder = self.create_bidi_encoder("context_encoder", embedding_dims, hidden_states)

        # bidirectional encoder for question
        question_encoder = self.create_bidi_encoder("question_encoder", embedding_dims, hidden_states)

        # Initialize the components (where not done upon creation)
        lookup.initialize()



        ###################
        # Wiring the components together
        #
        # Where present, the 3 letters at the end of the variable name identify its dimensions:
        # b ... position of the example within the batch
        # t ... position of the word within the document/question
        # f ... features of the embedding vector
        ###################

        ### Read the context document
        # Map token indices to word embeddings
        context_embedding_tbf = lookup.apply(context_bt.T)

        # Read the embedded context document using the bidirectional GRU and produce the contextual embedding of each word
        memory_encoded_btf = context_encoder.apply(context_embedding_tbf, context_mask_bt.T).dimshuffle(1,0,2)
        memory_encoded_btf.name = "memory_encoded_btf"

        ### Correspondingly, read the query
        x_embedded_tbf = lookup.apply(question_bt.T)
        x_encoded_btf = question_encoder.apply(x_embedded_tbf, question_mask_bt.T).dimshuffle(1,0,2)
        # The query encoding is a concatenation of the final states of the forward and backward GRU encoder
        x_forward_encoded_bf = x_encoded_btf[:,-1,0:hidden_states]
        x_backward_encoded_bf = x_encoded_btf[:,0,hidden_states:hidden_states*2]
        query_representation_bf = tt.concatenate([x_forward_encoded_bf,x_backward_encoded_bf],axis=1)

        # Compute the attention on each word in the context as a dot product of its contextual embedding and the query
        mem_attention_presoft_bt = tt.batched_dot(query_representation_bf, memory_encoded_btf.dimshuffle(0,2,1))

        # TODO is this pre-masking necessary?
        mem_attention_presoft_masked_bt = tt.mul(mem_attention_presoft_bt,context_mask_bt)

        # Normalize the attention using softmax
        mem_attention_bt = SoftmaxWithMask(name="memory_query_softmax").apply(mem_attention_presoft_masked_bt,context_mask_bt)

        if self.args.weighted_att:
            # compute weighted attention over original word vectors
            att_weighted_responses_bf = theano.tensor.batched_dot(mem_attention_bt, context_embedding_tbf.dimshuffle(1,0,2))


            # compare desired response to all candidate responses
            # select relevant candidate answer words
            candidates_embeddings_bfi = lookup.apply(candidates_bi).dimshuffle(0,2,1)

            # convert it to output symbol probabilities
            y_hat_presoft = tt.batched_dot(att_weighted_responses_bf, candidates_embeddings_bfi)
            y_hat = SoftmaxWithMask(name="output_softmax").apply(y_hat_presoft,candidates_bi_mask)

        else:
            # Sum the attention of each candidate word across the whole context document,
            # this is the key innovation of the model

            # TODO: Get rid of sentence-by-sentence processing?
            # TODO: Rewrite into matrix notation instead of scans?
            def sum_prob_of_word(word_ix, sentence_ixs, sentence_attention_probs):
                word_ixs_in_sentence = tt.eq(sentence_ixs,word_ix).nonzero()[0]
                return sentence_attention_probs[word_ixs_in_sentence].sum()

            def sum_probs_single_sentence(candidate_indices_i, sentence_ixs_t, sentence_attention_probs_t):
                result, updates = theano.scan(
                    fn=sum_prob_of_word,
                    sequences=[candidate_indices_i],
                    non_sequences=[sentence_ixs_t, sentence_attention_probs_t])
                return result

            def sum_probs_batch(candidate_indices_bt,sentence_ixs_bt, sentence_attention_probs_bt):
                result, updates = theano.scan(
                    fn=sum_probs_single_sentence,
                    sequences=[candidate_indices_bt, sentence_ixs_bt, sentence_attention_probs_bt],
                    non_sequences=None)
                return result

            # Sum the attention of each candidate word across the whole context document
            y_hat = sum_probs_batch(candidates_bi, context_bt, mem_attention_bt)
        y_hat.name = "y_hat"

        # We use the convention that ground truth is always at index 0, so the following are the target answers
        y = y.zeros_like()

        # We use Cross Entropy as the training objective
        cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
        cost.name = "cost"


        predicted_response_index = tt.argmax(y_hat,axis=1)
        accuracy = tt.eq(y,predicted_response_index).mean()
        accuracy.name = "accuracy"

        return cost, accuracy, mem_attention_bt, y_hat, context_bt, candidates_bi, candidates_bi_mask, y, context_mask_bt, question_bt, question_mask_bt


exp = ASReader()
exp.execute()