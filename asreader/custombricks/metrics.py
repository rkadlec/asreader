from blocks.bricks import application
from blocks.bricks.cost import Cost
import theano.tensor as tt

__author__ = 'rkadlec'


class CorrectResponseRank(Cost):
    """Calculates the the rank of a correct response among all competing responses given in the batch.
    It is assumed that the correct response is the first one in the original batch.
    """

    def __init__(self, examples_group_size=10):
        super(CorrectResponseRank, self).__init__()
        self.examples_group_size=examples_group_size

    @application(outputs=["correct_response"])
    def apply(self, y_hat):
        # reshape 1d vector to 2d matrix
        y_hat_2d = y_hat.reshape((y_hat.shape[0]/self.examples_group_size, self.examples_group_size))
        #y_hat_2d = tt.printing.Print("Y hat 2d in correct rank: ")(y_hat_2d)

        # sort each group by relevance
        # we sort the responses in decreasing order, that is why we multiply y_hat by -1
        sorting_indices = tt.argsort(-1 * y_hat_2d, axis=1)
        #sorting_indices = tt.printing.Print("sorting indices in correct rank: ")(sorting_indices)

        # check where is the ground truth whose index should be 0 in the original array
        correct_rank = tt.argmin(sorting_indices, axis=1) + 1
        #correct_rank = tt.printing.Print("correct rank: ")(correct_rank)
        correct_rank.name = "correct_rank"

        return correct_rank


class RecallAtN(Cost):
    """ Computes recall at n-th position based on rank of the correct answer computed by, e.g., CorrectResponseRank.
    """

    def __init__(self, n=1):
        super(RecallAtN, self).__init__()
        self.n = n

    def apply(self, correct_rank):
        #correct_rank = tt.printing.Print("correct rank: ")(correct_rank)
        recall_at_n = (correct_rank <= self.n).mean()
        #recall_at_n = tt.printing.Print("recall at " + str(self.n) + ": ")(recall_at_n)
        recall_at_n.name = "recall_at_" + str(self.n)
        return recall_at_n
