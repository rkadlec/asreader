# Creates CNN and DailzMail scripts from paper http://arxiv.org/abs/1506.03340
# these steps are opied from https://github.com/deepmind/rc-data

# Download Script
mkdir rc-data
cd rc-data
wget https://github.com/deepmind/rc-data/raw/master/generate_questions.py

# Download and Extract Metadata
wget https://storage.googleapis.com/deepmind-data/20150824/data.tar.gz -O - | tar -xz --strip-components=1

# Enter Virtual Environment and Install Packages
virtualenv venv
source venv/bin/activate
wget https://github.com/deepmind/rc-data/raw/master/requirements.txt
pip install -r requirements.txt

sudo apt-get install libxml2-dev libxslt-dev

# Download URLs
python generate_questions.py --corpus=dailymail --mode=download
python generate_questions.py --corpus=cnn --mode=download

# Generate Questions
python generate_questions.py --corpus=dailymail --mode=generate
python generate_questions.py --corpus=cnn --mode=generate

