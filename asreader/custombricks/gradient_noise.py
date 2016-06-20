from blocks.algorithms import StepRule
from blocks.roles import add_role, ALGORITHM_HYPERPARAMETER
from blocks.utils import shared_floatx
from theano.sandbox import rng_mrg
import theano
from collections import OrderedDict
from numpy import sqrt

__author__ = 'Ondrej Bajgar (OBajgar@cz.ibm.com)'


class GradientNoise(StepRule):
    '''
        Adds decaying gaussian noise to the gradient at each step as suggested in
        Neelakantan, Vilnis, Le, Sutskever, Kaiser, Kurach and Martens, "Adding Gradient Noise Improves Learning for
        Very Deep Networks", ICLR 2016

        Suggested values for eta: {0.01, 0.3, 1.0}
    '''

    def __init__(self, eta=0, gamma=0.55, seed=180891):

        self.eta_sqrt = shared_floatx(sqrt(eta), "eta")
        add_role(self.eta_sqrt, ALGORITHM_HYPERPARAMETER)

        self.gamma_half = shared_floatx(gamma/2, "gamma")
        add_role(self.gamma_half, ALGORITHM_HYPERPARAMETER)

        self.theano_random = rng_mrg.MRG_RandomStreams(seed=seed)

    def add_noise(self, x, t):
        sigma = self.eta_sqrt / (1+t) ** self.gamma_half
        noise = self.theano_random.normal(avg=0, std=sigma, size=x.shape, dtype=theano.config.floatX)
        x = x + noise
        return x

    def compute_steps(self, previous_steps):
        time = shared_floatx(0., 'time')

        t = time+1

        steps = OrderedDict(
            (parameter, self.add_noise(step, t))
            for parameter, step in previous_steps.items())

        return steps, [(time, t)]
