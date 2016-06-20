__author__ = 'kadlec'

import re
import os

def strip_param_name(param):
    if param.startswith("--"):
        return param[2:]
    elif param.startswith("-"):
        return param[1:]

def output_metaparams(output_file, metaparam2value):
    for (param,val) in metaparam2value.iteritems():
        output_file.write(strip_param_name(param) + "=" + val + "\n")
        output_file.write("\n")

def escapeBash(text):
    return text
    #return "\"" + text + "\""

def get_current_metaparams_str(parser, args):
    vals = extract_learning_metaparam_current_values(parser, args)
    str_list = [str(key) +"="+str(val).replace(" ", "_") for key,val in vals.iteritems()]
    return "_".join(str_list)

def extract_learning_metaparam_current_values(parser, args):
    metaparam2current_value = {}

    metaparam_re = re.compile(".*\(meta param suggested value: (.*)\).*")
    for argument in parser._positionals._actions:
        if argument.help:
            m = metaparam_re.match(argument.help)
            if m:
                metaparam = argument.option_strings[0]
                if metaparam[1] == '-':
                    metaparam = metaparam[2:]
                else:
                    metaparam = metaparam[1:]

                metaparam2current_value[metaparam] = getattr(args,argument.dest)

    return metaparam2current_value



def create_ariadne_config_skeleton(parser, version="1.0", root_folder="config"):
    """
    Creates default configuration files from description in script command line parameters.
    :param parser:
    :return:
    """

    # create the root folder if it does not exist
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)


    metaparam2value = {}

    metaparam_re = re.compile(".*\(meta param suggested value: (.*)\).*")
    for argument in parser._positionals._actions:
        if argument.help:
            m = metaparam_re.match(argument.help)
            if m:
                metaparam_value = m.group(1)
                metaparam2value[argument.option_strings[0]] = metaparam_value

    prog_name = parser.prog.split(".")[0]
    connector_script = prog_name + '_connector.sh'

    # create TSD
    tsd_params = ["$"+strip_param_name(param_name) for param_name in metaparam2value.iterkeys()]

    with open(os.path.join(root_folder,prog_name + ".tsd"), "w") as tsd_file:
        tsd_file.write("# Name of configuration\n")
        tsd_file.write(prog_name+"\n")
        tsd_file.write("\n")
        tsd_file.write("# Required definitions \n")
        tsd_file.write("Execution = ./" + connector_script + " \n")
        tsd_file.write("Parameters = @train_file @test_file " + " ".join(map(escapeBash,tsd_params)) + "\n")
        tsd_file.write("\n")
        tsd_file.write("Update = echo " + prog_name + "-" + version + "\n")
        tsd_file.write("\n")
        tsd_file.write("#======Default values======\n")
        output_metaparams(tsd_file, metaparam2value)
        tsd_file.write("\n")
        tsd_file.write("#==========================\n")



        #tsd_file.write("Purchase Amount: %s" % TotalAmount)

    # create TST
    with open(os.path.join(root_folder,prog_name + ".tst"), "w") as tst_file:
        tst_file.write("# Name of configuration\n")
        tst_file.write(prog_name+"\n\n")
        tst_file.write("Algorithm=gp_adaptive_max(2,accuracy)\n\n")

        output_metaparams(tst_file, metaparam2value)

    # create connector script for Bivoj
    with open(os.path.join(root_folder,prog_name + "_connector.sh"), "w") as con_file:

        # preprocess input vars
        offset = 3
        #for i in xrange(offset,len(metaparam2value)+offset):
        #    i = str(i)
        #    con_file.write(i + "=${" + i + "#\\\"}\n")
        #    con_file.write(i + "=${" + i + "%\\\"}\n")

        con_file.write("python $PCD/blocks-nlp/"+prog_name+".py --train $1 --valid $2 ")
        # all double quotes are removed from the variables using BASH ${var//\"} construct
        #param_binding = [param+" ${"+ str(i+3)+"#\\\"}" for (i, param) in enumerate(metaparam2value.iterkeys())]
        param_binding = [param+" ${"+ str(i+3)+"}" for (i, param) in enumerate(metaparam2value.iterkeys())]
        con_file.write(" ".join(param_binding))

        # add embeddings root pointing to Bivoj dedicated dir
        con_file.write(" --embeddings_root $PCD/../../word_models")
        con_file.write(" --disable_progress_bar")

        con_file.write("\n")


    # create info.json for Bivoj
    with open(os.path.join(root_folder,"info.json"), "w") as info_file:
        info_file.write('{\n')
        info_file.write('  "dataDefinition": "string_label",\n')
        info_file.write('  "name": "' + prog_name +'",\n')
        info_file.write('  "version": "' + version + '",\n')
        info_file.write('  "trainFileReplace": "@train_file",\n')
        info_file.write('  "testFileReplace": "@test_file",\n')
        info_file.write('  "arffMetadata": {\n')
        info_file.write('    "columns": [\n')
        info_file.write('      {\n')
        info_file.write('        "role": "FEATURE",\n')
        info_file.write('        "type": "STRING",\n')
        info_file.write('        "name": "text"\n')
        info_file.write('      },\n')
        info_file.write('      {\n')
        info_file.write('        "role": "LABEL",\n')
        info_file.write('        "type": "STRING",\n')
        info_file.write('        "name": "label"\n')
        info_file.write('      }\n')
        info_file.write('    ]\n')
        info_file.write('  },\n')
        info_file.write('  "connector": "' + connector_script + '",\n')
        info_file.write('  "singleLabel": true,\n')
        info_file.write('  "givesLabels": false\n')
        info_file.write('}\n')

    print "Config created in " + root_folder + " directory"