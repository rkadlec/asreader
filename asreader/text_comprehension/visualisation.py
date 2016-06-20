__author__ = 'schmidm'


import random

def make_html_file(examples, filename):
    """
    Visualizes attention of a model in a HTML file.
    :param examples:
    :param filename:
    :return:
    """


    def attention_to_rgb(attention):
        # red = int(attention * 255)
        # green = int(255 - red)

        red = 255
        green = int(255 * (1-attention))
        blue = int(255 * (1-attention))

        return 'rgb(%s,%s,%s)' % (str(red), str(green), blue)


    out_file = open(filename, 'w')
    out_file.write('''<!DOCTYPE html>
                    <head>
                    <link rel="stylesheet" href="//code.jquery.com/ui/1.10.4/themes/smoothness/jquery-ui.css" />
                    <script src="http://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
                    <script src="http://code.jquery.com/ui/1.10.4/jquery-ui.js"></script>
                    </head><html>
                    <body><h1>Awesome Network Output</h1>
                    <script>

                    function highlight(x)
                    {
                        //alert(x)
                        $("." + x).addClass('foo')
                        //$(".foo").css({'font-weight': 600})
                        $(".foo").animate({
                            color: "blue"
                            }, {duration: 200} );
                    }

                    function unhighlight()
                    {
                        $(".foo").css({'font-weight': "normal"})
                        $(".foo").animate({
                            color: "black"
                            }, {duration: 200} );
                        $(".foo").removeClass('foo')
                    }
                    </script>
                    ''')


    #1.0 sort answers by how well we predicted them - the fifth entry is the correct answer index
    examples = sorted(examples, key=lambda example: example[5])

    for example_index, example in enumerate(examples):

        question, context_words, context_attention, answers, answers_attention, correct_answer_idnex = example

        out_file.write("<h2>%(example_index)s</h2>" %  {'example_index' : correct_answer_idnex})
        out_file.write("<p>")
        for word, attention in zip(context_words, context_attention):
            if(word in answers):
                out_file.write('<u>')

            out_file.write('<mark class="g%(class)s" cursor="pointer" title="%(pointer_text)s" style="background-color:%(rgb)s;">%(word)s </mark>' %
                           {'pointer_text' : str(attention), 'rgb' : attention_to_rgb(attention), 'word' : word, 'class' : str(example_index) + "-" + word.replace("@", "")})

            #str(example_index) + "." + word

            if(word in answers):
                out_file.write('</u>')
        out_file.write("</p>")

        out_file.write('<p>%(question)s</p>' % {'question' : question})

        out_file.write("<p>")
        for i, (answer, attention) in enumerate(zip(answers, answers_attention)):
            answer = answer.replace("<", "").replace(">", "")
            out_file.write('<mark onmouseleave=unhighlight() onmouseover=highlight("g%(class)s") cursor="pointer" title="%(pointer_text)s" style="background-color:%(rgb)s;">%(index)s ) %(word)s; </mark>' %
                            {'index' : i, 'pointer_text' : str(attention), 'rgb' : attention_to_rgb(attention), 'word' : answer, 'class' : str(example_index) + "-" + answer.replace("@", "")})
        out_file.write("</p>")


    out_file.write('</body></html>')
    out_file.close()



if __name__ == "__main__":

    examples = []

    for i in range(4):
        question = "Who's the best?"
        context = 'Ruda met Alice . Bob met Ruda . Alice met Ruda .'.split(' ')
        context_attention = map(lambda x : random.uniform(0, 1), context)
        answers = 'Bob Steve Alice'.split(' ')
        answers_prob = map(lambda x : random.uniform(0, 1), answers)
        correct_answer_idnex = random.uniform(0, 10)

        examples.append((question, context, context_attention, answers, answers_prob, correct_answer_idnex))

    make_html_file(examples, 'output.html')