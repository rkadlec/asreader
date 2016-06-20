# Attention Sum Reader

## Introduction

This is a Theano/Blocks implementation of the Attention Sum Reader model as presented in "Text Comprehension with the Attention Sum Reader Network" available at http://arxiv.org/abs/1603.01547.
We encourage you to familiarize yourself with the model by reading the above article prior to studying the particulars of this implementation.

# Quick start
If you want to get started as fast as possible try this:
```
./prerequisites.sh
cd asreader
./quick-start-cbt-ne.sh
```
If you do not have a GPU available, remove the device=gpu flag from quick-start-generic.sh. However note that training the text comprehension tasks on a CPU is likely to take a prohibitively long time.


This should install the prerequisites, download the CBT dataset, train two models on the named-entity part of the data, form an ensemble and report the accuracies.


## License

© Copyright IBM Corporation. 2016.

This licensed material is licensed for academic, non-commercial use only . The licensee may use, copy, and modify the licensed materials in any form without payment to IBM for the sole purposes of evaluating and extending the licensed materials for non-commercial purposes.
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the licensed materials..

Notwithstanding anything to the contrary, IBM PROVIDES THE LICENSED MATERIALS ON AN "AS IS" BASIS AND IBM DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES OR CONDITIONS OF MERCHANTABILITY, SATISFACTORY QUALITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, AND ANY WARRANTY OR CONDITION OF NON-INFRINGEMENT. IBM SHALL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY OR ECONOMIC CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR OPERATION OF THE LICENSED MATERIALS. IBM SHALL NOT BE LIABLE FOR LOSS OF, OR DAMAGE TO, DATA, OR FOR LOST PROFITS, BUSINESS REVENUE, GOODWILL, OR ANTICIPATED SAVINGS. IBM HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS OR MODIFICATIONS TO THE LICENSED MATERIALS. 
                                                                                                         


# Detailed usage

# Installation

## Quick installation
Provided you have python with pip installed, running
```
prerequisites.sh
```
should install Blocks and dependencies for you.
It also downloads the Children's Book Test dataset and the CNN and Daily Mail news datasets. We are aware that
the news data download sometimes crashes. Rerunning the script prepare-rec-data.sh should be able to resume
the download if that happens (alrenatively you can download the datasets at TODO).

However if you prefer to install the dependencies by yourself, some details are below:

## Dependencies:
1. HDF5 (required for installing Blocks)
    In the Debian/Ubuntu family of distributions, you should be able to install the library using
    ```
    sudo apt-get install libhdf5-serial-dev
    ```
    Otherwise installation instructions and source download can be found at http://hdfgroup.org/HDF5/release/obtain5.html
2. Blocks and its dependencies
    Installation instructions can be found at blocks.readthedocs.io/en/latest/setup.html.
    You should be able to install Blocks including Theano and other dependencies using pip by TODO

3. NLTK + punkt corpus (TODO is this needed?)
   This tokenizer that we use for reading the bAbI datasets can be installed using
   ```
   pip install nltk --user
   python -m nltk.downloader punkt
    ```



## Getting data
### Children Book Test
Children Book Test data should be already downloaded by the quick start script. If you skipped this script you can prepare the data by `prepare-cbt-data.sh`

### CNN and Daily Mail
The best way how to get the CNN and DailyMail datasets is to download the questions and stories files from http://cs.nyu.edu/~kcho/DMQA/.
Place them into folder `$CLONE_DIR/data` and run a script `$CLONE_DIR/data/prepare-rc-data-downloaded.sh`.

Alternatively you can use a script `$CLONE_DIR/data/prepare-rc-data.sh` that downloads the data using the original scripts from https://github.com/deepmind/rc-data. However,
the news data download sometimes crashes. Therefore it is often necessary to download missing articles by re-running `generate_questions.py` script.

Now when you have CNN and DailyMail datasets you can use them to train the models:
```
cd asreader
./quick-start-cnn.sh
./quick-start-dm.sh
```

## Training models
The model can be trained by running the `text-comprehension/as_reader.py` script. The simplest usage is:
```
python text_comprehension/as_reader.py --dataset_root data/CBTest/data/ --train train.txt --valid valid.txt --test test.txt
```
where the .txt files are the appropriate datasets. Some of the recommended configurations can be copied from the quick-start-cbt-ne.sh script

You may need to prepend the following prefixes in front of the command and run it from the project root directory
```
THEANO_FLAGS="floatX=float32,device=gpu" PYTHONPATH="$PYTHONPATH:./"
```
Some of the most useful command line arguments you may wish to use are the following
* --dataset_type [cbt|cnn|babi] - the type of dataset that is being used. Defaults to the Children's Book Test
* -b 32 ... batch size - larger values usually speed up training however increase the memory usage
* -sed 256 ... source embedding dimension
* -ehd 256 ... the number of hidden units in each half of the bidirectional GRU encoders
* -lr 0.001 ... learning rate
* --output_dir ... output directory for the validation and test prediction files
* --patience_metric accuracy ... when this metric stops improving, training is eventually stopped
* -p 1 ... the number of epochs for which training continues since achieving the best value of the patience metric
* --own_eval ... runs a script that eb
* --append_metaparams ... includes the metaparameters in the filename of the generated prediction files - useful when generating multiple models
* --weighted_att ... instead of attention sum, use the weighted attention model to which we compare the ASReader in the paper
The full list of parameters with descriptions can be displayed by running the script with the -h flag.


## Ensembling
`as_reader.py` can generate the predictions for the test and validation datasets into the output directory. By default the predictions are generated every epoch. The text_comprehension/eval/copyBestPredictions directory can then be used to find the time at which model achieved the best validation accuracy and it copies the corresponding validation and test predictions to a separate folder.
An example syntax is
```
python text_comprehension/eval/copyBestPredictions.py -vp cbtest_NE_valid_2000ex.txt. -tp cbtest_NE_test_2500ex.txt. -i out_dir -o out_dir/best_predictions
```
where `-vp` and `-tp` give the prefixes of the validation and test predictions respectively. These are usually the validation and test dataset filenames.

Once the `best_predictions` directory contains only one test and one validation prediction for each model, we can fuse these using the `text_comprehension/eval/fusion.py` for instance using the following command:
```
python text_comprehension/eval/fusion.py -pr "out_dir/best_predictions/*.y_hat_valid" -o $OUT_DIR/best_predictions/simple_fusion.y_hat -t foo --fusion_method AverageAll
```
where `-pr` gives an expression for the validation predictions to be used and `-o` specifies the file to output.

The script provides three methods of fusion toggled by the `--fusion_method` parameter:
* `AverageAll` - the ensemble prediction is a mean of all the supplied single-model predictions
* `pBest` - sorts the candidate models by validation accuracy and selects the best proportion p of models to form the ensemble
* `AddImprover` - sorts the candidate models by validation accuracy and then tries adding them to the ensemble in that order
              keeping each model in the ensemble only if it improves its val. accuracy


# Contributors
Rudolf Kadlec, Martin Schmid, Ondrej Bajgar, Tamir Klinger, Ladislav Kunc, Jan Kleindienst
