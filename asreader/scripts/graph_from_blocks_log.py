__author__ = 'rkadlec'

import argparse
import re
from itertools import cycle

import matplotlib.pyplot as plt

"""
Draws a graph from Blocks training text log (printed to stdout).

"""

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="Utility that plots training progress stored in stdout output of Blocks training log.")

parser.add_argument('-m','--metric_to_plot', nargs='+', default="valid_cost",
                    help='name of a metrics to plot')

parser.add_argument('-f', '--file', nargs='+',
                    help='log files to plot in a graph')

args = parser.parse_args()


def load_text_file(filename, logged_value):
    """
    Loads values from a single training log.
    :param filename:
    :param logged_value:
    :return:
    """
    pattern = re.compile(".* " + logged_value + ": (.*)")
    end_pattern = re.compile("Sender: LSF System.*")
    values = []
    with open(filename) as f:
        for line in f:
            end = end_pattern.match(line)
            if end: break 
            m = pattern.match(line)
            if m:
                # pick the value from regexp
                values.append(float(m.group(1)))
    #print (values)
    return values



def load_multiple_values_from_multiple_files(files_list, logged_values):
    out = dict()
    for logged_value in logged_values:
        loaded_values = load_values_from_multiple_files(files_list, logged_value)
        #out.update(loaded_values)
        out[logged_value] = loaded_values
    # strip dirs in filepaths
    return out



def load_values_from_multiple_files(files_list, logged_value):
    loaded_values = map(lambda x: load_text_file(x, logged_value), files_list)
    # strip dirs in filepaths
    filenames = [logged_value+"_"+x.split("/")[-1] for x in files_list]
    lines_dict = dict(zip(filenames, loaded_values))

    # print table sorted by extreme values
    name_and_max = map(lambda x: (x[0], max([0] + x[1])),lines_dict.iteritems())
    name_and_max = sorted(name_and_max, key=lambda x: x[1])

    print
    print "{} best results".format(logged_value)
    for pair in name_and_max:
        print pair

    return lines_dict


lines_dict = load_multiple_values_from_multiple_files(args.file, args.metric_to_plot)


fig = plt.figure()


# line stzles
lines = ["-","--","-."]
linecycler = cycle(lines)


for metric, lines in lines_dict.iteritems():
    # add all lines to a plot
    style = next(linecycler)
    for name, sequence in lines.iteritems():
        plt.plot(sequence, label=name, picker=3, linestyle=style)


# mouse hover event handling function
def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    print "Label: " + thisline._label
    print 'onpick points:', zip(xdata[ind], ydata[ind])

fig.canvas.mpl_connect('pick_event', onpick)


lgd = plt.legend(bbox_to_anchor=(0.5, -0.1), loc=9, borderaxespad=0.)
fig.savefig('samplefig.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
#fig.savefig('samplefig.png', additional_artists=[lgd], bbox_inces='tight')

plt.show()
