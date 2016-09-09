# HDF5 is a prerequisite for installing Blocks (uncomment the following if you have apt on your system and root privileges)
# sudo apt-get install libhdf5-serial-dev

# Install Theano+Blocks and their dependencies (remove --user to install for all users (requires root privileges))
pip install git+http://github.com/mila-udem/blocks.git@359afad119f8c6ac0ebc3cc6ec6e6475656babae -r https://raw.githubusercontent.com/mila-udem/blocks/master/requirements.txt --user
# nltk tokenizer + punkt corpus (used for tokenizing the bAbI datasets)
pip install nltk --user
python -m nltk.downloader punkt

# prepare the Children's Book Test dataset
cd data
./prepare-cbt-data.sh

# CNN and DailyMail data are not prepared by this default script since it takes some time to process them and
# they also have to be downloaded manually, see README.md
#./prepare-rc-data-downloaded.sh
