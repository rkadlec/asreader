__author__ = 'lada'

import os.path

from blocks.extensions import SimpleExtension
from blocks.serialization import secure_dump

SAVED_TO = "saved_to"


class SaveTheBest(SimpleExtension):
    """Check if a log quantity has the minimum/maximum value so far
    and if that is true then save a pickled version of the main loop
    to the disk.

    The pickled main loop can be later reloaded and training can be
    resumed.
    Makes a `SAVED_TO` record in the log with the serialization destination
    in the case of success and ``None`` in the case of failure. The
    value of the record is a tuple of paths to which saving was done
    (there can be more than one if the user added a condition
    with an argument, see :meth:`do` docs).

    Parameters
    ----------
    record_name : str
        The name of the record to track.
    choose_best : callable, optional
        A function that takes the current value and the best so far
        and return the best of two. By default :func:`min`, which
        corresponds to tracking the minimum value.
    path : str
        The destination path for pickling.
    save_separately : list of str, optional
        The list of the main loop's attributes to be pickled separately
        to their own files. The paths will be formed by adding the
        attribute name preceded by an underscore before the before the
        `path` extension. The whole main loop will still be pickled
        as usual.
    use_cpickle : bool
        See docs of :func:`~blocks.serialization.dump`.

    Attributes
    ----------
    best_name : str
        The name of the status record to keep the best value so far.


    Notes
    -----
    Using pickling for saving the whole main loop object comes with
    certain limitations:

    * Theano computation graphs build in the GPU-mode cannot be used in
    the usual mode (and vice-versa). Therefore using this extension
    binds you  to using only one kind of device.

    """
    def __init__(self, record_name, path, choose_best=min,
                 save_separately=None, use_cpickle=False, **kwargs):
        self.record_name = record_name
        self.best_name = "bestsave_" + record_name
        self.choose_best = choose_best
        if not save_separately:
            save_separately = []
        self.path = path
        self.save_separately = save_separately
        self.use_cpickle = use_cpickle
        # kwargs.setdefault("after_training", True)
        kwargs.setdefault("after_epoch", True)
        super(SaveTheBest, self).__init__(**kwargs)

    def save_separately_filenames(self, path):
        """ Compute paths for separately saved attributes.

        Parameters
        ----------
        path : str
            Path to which the main savethebest file is being saved.

        Returns
        -------
        paths : dict
            A dictionary mapping attribute names to derived paths
            based on the `path` passed in as an argument.
        """
        root, ext = os.path.splitext(path)
        return {attribute: root + "_" + attribute + ext
                for attribute in self.save_separately}

    def do(self, which_callback, *args):
        current_value = self.main_loop.log.current_row.get(self.record_name)
        if current_value is None:
            return
        best_value = self.main_loop.status.get(self.best_name, None)
        if(best_value is None or
                (current_value != best_value and
                    self.choose_best(current_value, best_value) == current_value)):
            self.main_loop.status[self.best_name] = current_value
            # save main_loop
            _, from_user = self.parse_args(which_callback, args)
            try:
                path = self.path
                if from_user:
                    path, = from_user
                secure_dump(self.main_loop, path, use_cpickle=self.use_cpickle)
                filenames = self.save_separately_filenames(path)
                for attribute in self.save_separately:
                    secure_dump(getattr(self.main_loop, attribute),
                                filenames[attribute], use_cpickle=self.use_cpickle)
            except Exception:
                path = None
                raise
            finally:
                already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
                self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                            (path,))

import logging
logger = logging.getLogger(__name__)