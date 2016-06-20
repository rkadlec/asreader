__author__ = 'martin.schmid@cz.ibm.com'

from cbt_dataset import CBDataset

'''
    This module serves for reading the CNN and Daily Mail datasets as introduced in http://arxiv.org/abs/1603.01547
    into the ASReader model.
'''

class CNNDataset(CBDataset):
    def __init__(self, file, dictionary, bos_token='<S>', eos_token='</S>', unk_token='<UNK>', level='word',
                 preprocess=None, append_question=False, question_end_token='<QUESTION_END>',
                 add_attention_features=False):

        super(CNNDataset, self).__init__(file, dictionary, bos_token, eos_token, unk_token, level, preprocess,
                                         append_question, question_end_token, add_attention_features)

    def get_data(self, state=None, request=None):
        if request is not None:
            raise ValueError
        lines = []

        line = next(state).strip()
        while line != "##########":
            lines.append(line)
            line = next(state).strip()

        story = lines[2]    # Context document
        question = lines[4] # Query
        answer = lines[6]   # Correct answer

        candidates_list = lines[8:] # Answer candidates

        # Move the correct answer to the first position among the candidates
        candidates_list.remove(answer)
        candidates_list.insert(0, answer)

        candidates_strs = " ".join(candidates_list)

         # Add the question at the beginning and end of the context document to direct the context encoder
        if self.append_question:
            story = question + " " + self.question_end_token + " " + story + " " + self.question_end_token + " " + question

        return (self.translate_one_line(story), self.translate_one_line(question),
                self.translate_one_line(answer), self.translate_one_line(candidates_strs),)