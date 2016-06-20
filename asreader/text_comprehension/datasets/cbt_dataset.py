from picklable_itertools import iter_, chain

from fuel.datasets import Dataset

class CBDataset(Dataset):
    r"""Reads text files and numberizes them given a dictionary.

    Parameters
    ----------
    files : list of str
        The names of the files in order which they should be read. Each
        file is expected to have a sentence per line.
    dictionary : str or dict
        Either the path to a Pickled dictionary mapping tokens to integers,
        or the dictionary itself. At the very least this dictionary must
        map the unknown word-token to an integer.
    bos_token : str or None, optional
        The beginning-of-sentence (BOS) token in the dictionary that
        denotes the beginning of a sentence. Is ``<S>`` by default. If
        passed ``None`` no beginning of sentence markers will be added.
    eos_token : str or None, optional
        The end-of-sentence (EOS) token is ``</S>`` by default, see
        ``bos_taken``.
    unk_token : str, optional
        The token in the dictionary to fall back on when a token could not
        be found in the dictionary.
    level : 'word' or 'character', optional
        If 'word' the dictionary is expected to contain full words. The
        sentences in the text file will be split at the spaces, and each
        word replaced with its number as given by the dictionary, resulting
        in each example being a single list of numbers. If 'character' the
        dictionary is expected to contain single letters as keys. A single
        example will be a list of character numbers, starting with the
        first non-whitespace character and finishing with the last one.
    preprocess : function, optional
        A function which takes a sentence (string) as an input and returns
        a modified string. For example ``str.lower`` in order to lowercase
        the sentence before numberizing.

    Examples
    --------
    >>> with open('sentences.txt', 'w') as f:
    ...     _ = f.write("This is a sentence\n")
    ...     _ = f.write("This another one")
    >>> dictionary = {'<UNK>': 0, '</S>': 1, 'this': 2, 'a': 3, 'one': 4}
    >>> def lower(s):
    ...     return s.lower()
    >>> text_data = TextFile(files=['sentences.txt'],
    ...                      dictionary=dictionary, bos_token=None,
    ...                      preprocess=lower)
    >>> from fuel.streams import DataStream
    >>> for data in DataStream(text_data).get_epoch_iterator():
    ...     print(data)
    ([2, 0, 3, 0, 1],)
    ([2, 0, 4, 1],)

    .. doctest::
       :hide:

       >>> import os
       >>> os.remove('sentences.txt')

    """
    provides_sources = ('context', 'question', 'answer', 'candidates',)
    #provides_sources = ('context', 'question', 'answer',)
    example_iteration_scheme = None

    def __init__(self, files, dictionary, bos_token='<S>', eos_token='</S>',
                 unk_token='<UNK>', level='word', preprocess=None, append_question=False, question_end_token='<QUESTION_END>',
                 add_attention_features=False):
        self.files = files
        self.dictionary = dictionary
        if bos_token is not None and bos_token not in dictionary:
            raise ValueError
        self.bos_token = bos_token
        if eos_token is not None and eos_token not in dictionary:
            raise ValueError
        self.eos_token = eos_token
        if unk_token not in dictionary:
            raise ValueError
        self.unk_token = unk_token
        if level not in ('word', 'character'):
            raise ValueError
        self.level = level
        self.preprocess = preprocess
        self.append_question = append_question
        self.question_end_token = question_end_token
        self.add_attention_features = add_attention_features


        super(CBDataset, self).__init__()

    def open(self):
        return chain(*[iter_(open(f)) for f in self.files])

    def translate_one_line(self, sentence):

        if self.preprocess is not None:
            sentence = self.preprocess(sentence)
        data = [self.dictionary[self.bos_token]] if self.bos_token else []
        if self.level == 'word':
            data.extend(self.dictionary.get(word,
                                            self.dictionary[self.unk_token])
                        for word in sentence.split())
        else:
            data.extend(self.dictionary.get(char,
                                            self.dictionary[self.unk_token])
                        for char in sentence.strip())
        if self.eos_token:
            data.append(self.dictionary[self.eos_token])

        return data

    def get_data(self, state=None, request=None):
        if request is not None:
            raise ValueError
        lines = []

        sentence = next(state).strip()
        while sentence:
            lines.append(sentence)
            sentence = next(state).strip()

        # translate context
        context_str = " ".join(lines[:-1])

        question_str, answer_str, _, candidates_tmp = lines[-1].split("\t")
        candidates_list = candidates_tmp.split("|")[:10] # use the first 10 candidates

        # add question before and after the context
        if self.append_question:
            context_str = question_str + " " + self.question_end_token + " " + context_str + " " + self.question_end_token + " " + question_str

        # replace spaces with UNK
        for n,str in enumerate(candidates_list):
            if str is "":
                candidates_list[n]=self.unk_token

        candidates_list.remove(answer_str)
        candidates_list.insert(0,answer_str)
        candidates_strs = " ".join(candidates_list)


        """
        tmp = self.translate_one_line(candidates_strs)
        if len(tmp) <> 10:
            print "ERR"
            print candidates_tmp
        """

        return (self.translate_one_line(context_str), self.translate_one_line(question_str),
                self.translate_one_line(answer_str), self.translate_one_line(candidates_strs),)


