# unpacks CNN and Daily Mail datasets

# unpack all files
tar -zxvf cnn.tgz
tar -zxvf cnn_stories.tgz
#tar -zxvf dailymail.tgz
#tar -zxvf dailymail_stories.tgz

function process_dir_fn {
     ../../rc-convert.sh questions/validation validation.txt
     ../../rc-convert.sh questions/training training.txt
     ../../rc-convert.sh questions/test test.txt
}

echo "Processing CNN dataset"
cd cnn
process_dir_fn

echo "Processing DailyMail dataset"
cd ../dailymail
process_dir_fn




