
"""
Tim Klinger's code for masked softmax.
"""
from blocks.bricks import Softmax
from theano import tensor
from blocks.bricks.wrappers import WithExtraDims
from blocks.bricks.base import application

class SoftmaxWithMask(Softmax):

    @application(inputs=['input_', 'mask_'], outputs=['output'])
    def log_probabilities(self, input_, mask_):
        """Normalize log-probabilities.

        Converts unnormalized log-probabilities (exponents of which do not
        sum to one) into actual log-probabilities (exponents of which sum
        to one).

        Parameters
        ----------
        input_ : :class:`~theano.Variable`
            A matrix, each row of which contains unnormalized log-probabilities of a
            distribution.
        mask_: a matrix, each row of which contains a 1 if the value is to be included in the
            normalization and 0 otherwise

        Returns
        -------
        output : :class:`~theano.Variable`
            A matrix with normalized log-probabilities in each row for each
            distribution from `input_` or 0 if the entry is masked

        """
        input_masked = mask_ * input_
        shifted = mask_ * (input_masked - input_masked.max(axis=1, keepdims=True))
        Z = tensor.log((mask_ * tensor.exp(shifted)).sum(axis=1, keepdims=True))
        result = mask_ * (shifted - Z)

        # DEBUG
        """
        input_masked = theano.printing.Print("input_masked: ")(input_masked)
        shifted = theano.printing.Print("shifted: ")(shifted)
        Z = theano.printing.Print("Z: ")(Z)
        result = theano.printing.Print("result: ")(result)
        """

        return result

    @application(inputs=['input_','mask'], outputs=['output'])
    def apply(self, input_, mask):
        """Standard softmax.

        Parameters
        ----------
        input_ : :class:`~theano.Variable`
            A matrix, each row contains unnormalized log-probabilities of a
            distribution.

        Returns
        -------
        output_ : :class:`~theano.Variable`
            A matrix with probabilities in each row for each distribution
            from `input_`.

        """
        #return tensor.nnet.softmax(input_)
        return mask * tensor.exp(self.log_probabilities(input_,mask))

    @application(inputs=['y', 'x', 'mask'], outputs=['output'])
    def categorical_cross_entropy_with_masking(self, application_call, y, x, mask, **kwargs):
        """Computationally stable cross-entropy for pre-softmax values.

        Parameters
        ----------
        y : :class:`~tensor.TensorVariable`
            In the case of a matrix argument, each row represents a
            probabilility distribution. In the vector case, each element
            represents a distribution by specifying the position of 1 in a
            1-hot vector.
        x : :class:`~tensor.TensorVariable`
            A matrix, each row contains unnormalized probabilities of a
            distribution.
        mask: a mask of the elements to filter in the same shape as x

        Returns
        -------
        cost : :class:`~tensor.TensorVariable`
            A vector of cross-entropies between respective distributions
            from y and x.

        """
        x = self.log_probabilities(x, mask)
        # DEBUG
        #x = theano.tensor.printing.Print("log probabilities: ")(x)
        #y = theano.tensor.printing.Print("target factoids: ")(y)
        #mask = theano.tensor.printing.Print("mask: ")(mask)
        application_call.add_auxiliary_variable(
            x.copy(name='log_probabilities'))
        if y.ndim == x.ndim - 1:
            indices = tensor.arange(y.shape[0]) * x.shape[1] + y
            cost = -x.flatten()[indices]
        elif y.ndim == x.ndim:
            cost = -(x * y).sum(axis=1)
        else:
            raise TypeError('rank mismatch between x and y')
        return cost

class NDimensionalSoftmaxWithMask(SoftmaxWithMask):
    decorators=[WithExtraDims()]