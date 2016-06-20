from collections import OrderedDict

from fuel.datasets import IndexableDataset, Dataset
from fuel.schemes import SequentialExampleScheme, ShuffledExampleScheme


__author__ = 'rkadlec'


class UnpickableIndexableDataset(Dataset):
    """Creates a dataset from a set of indexable containers. Data stored in this dataset won't be stored when the model
    is serialized.

    Parameters
    ----------
    indexables : :class:`~collections.OrderedDict` or indexable
        The indexable(s) to provide interface to. This means it must
        support the syntax ```indexable[0]``. If an
        :class:`~collections.OrderedDict` is given, its values should be
        indexables providing data, and its keys strings that are used as
        source names. If a single indexable is given, it will be given the
        source ``data``.

    Attributes
    ----------
    indexables : list
        A list of indexable objects.

    Notes
    -----
    If the indexable data is very large, you might want to consider using
    the :func:`.do_not_pickle_attributes` decorator to make sure the data
    doesn't get pickled with the dataset, but gets reloaded/recreated
    instead.

    This dataset also uses the source names to create properties that
    provide easy access to the data.

    """
    def __init__(self, indexables, start=None, stop=None, **kwargs):
        if isinstance(indexables, dict):
            self.provides_sources = tuple(indexables.keys())
        else:
            self.provides_sources = ('data',)
        super(UnpickableIndexableDataset, self).__init__(**kwargs)
        if isinstance(indexables, dict):
            self.indexables = [indexables[source][start:stop]
                               for source in self.sources]
            if not all(len(indexable) == len(self.indexables[0])
                       for indexable in self.indexables):
                raise ValueError("sources have different lengths")
        else:
            self.indexables = [indexables]

        self.example_iteration_scheme = SequentialExampleScheme(self.num_examples)

        self.start = start
        self.stop = stop

    def __getstate__(self):
        """
        This function is called when Blocks want to serialize the model. This implementation prevents the data from being
        stored.
        :return:
        """
        d = dict(self.__dict__)
        del(d['indexables'])
        return d

    @property
    def num_examples(self):
        return len(self.indexables[0])

    def get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError
        return tuple(indexable[request] for indexable in self.indexables)


