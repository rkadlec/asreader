from collections import Counter
import cPickle as pickle
import nltk

__author__ = 'rkadlec'


def tokenize_on_whitespace(text):
    tokens = text.split()
    return tokens


def default_sentence_to_tokens(text):
    tokens = nltk_token_sentence_to_tokens(text)
    return tokens


def nltk_token_sentence_to_tokens(text):
    tokens = nltk.word_tokenize(text)
    return tokens


def get_vocabulary(input_stream, sentence_to_tokens_fn=None, progress_indication_lines=100000):
    """
    Computes vocabulary sorted by word frequency.
    :param text_file:
    :param preprocess:
    :return:
    """

    word_counts = Counter()

    if not sentence_to_tokens_fn:
        sentence_to_tokens_fn = default_sentence_to_tokens

    line_counter = 0

    for line in input_stream:
        tokens = sentence_to_tokens_fn(line.rstrip('\n'))
        word_counts.update(tokens)

        line_counter += 1
        if line_counter % progress_indication_lines == 0:
            print "Processed line " + str(line_counter)


    # summary statistics
    total_words = sum(word_counts.values())
    distinct_words = len(list(word_counts))


    print "STATISTICS"
    print "Total words: " + str(total_words)
    print "Total distinct words: " + str(distinct_words)

    return word_counts


def get_most_frequent_words(text_file, number_of_words, sentence_to_tokens_fn=None):
    """
    Computes top N most frequent words from a text file.
    :param text_file:
    :param number_of_words:
    :param preprocess:
    :return:
    """
    word_counts = get_vocabulary(open(text_file), sentence_to_tokens_fn)

    top_n = word_counts.most_common(number_of_words)

    # summary statistics
    total_words = sum(word_counts.values())
    words_covered_by_top_n = sum(top_n.values())

    print "Percentage of words covered by top " + str(number_of_words) + " words: " + str(words_covered_by_top_n/total_words)

    return dict(top_n).keys()

def load_vocabulary(vocabulary_file):
    vocabulary = pickle.load(open(vocabulary_file))
    code2token = map(lambda x : x[0], vocabulary.most_common())
    return vocabulary, code2token, compute_token2code(code2token), compute_code2frequency(vocabulary, code2token)

def compute_token2code(code2word):
    return {v: i for i, v in enumerate(code2word)}

def compute_code2frequency(vocabulary_counter, code2word):
    return [vocabulary_counter[token] for token in code2word]

