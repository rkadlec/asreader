DATA_ROOT=$1
TRAIN=$2
VALID=$3
TEST=$4
OUT_DIR=$5
DATASET_TYPE=$6


# The datasets should be already downloaded and preprocessed

# Train a few models
for HYPERPARAMETERS in "${@:7}"
do
    LOG_FILE=$TRAIN.${HYPERPARAMETERS// /_}.log
    mkdir $OUT_DIR -p
    THEANO_FLAGS="floatX=float32,device=gpu" PYTHONPATH="$PYTHONPATH:$BASH_SOURCE" python text_comprehension/as_reader.py -b 32 $HYPERPARAMETERS -lr 0.001 -p 2 --dataset_root $DATA_ROOT --train $TRAIN --valid $VALID --test $TEST --output_dir $OUT_DIR --evaluate_every_n 1e --patience_metric accuracy --own_eval --append_metaparams --no_html --dataset_type $DATASET_TYPE --save_path $OUT_DIR/model.blocks.pkl | tee $OUT_DIR/$LOG_FILE -i
done

# For each model select the best validation prediction (among the ones saved for each epoch) and copy it to a separate directory:
# along with the corresponding test prediciton:
python text_comprehension/eval/copyBestPredictions.py -vp validation.txt. -tp test.txt. -i $OUT_DIR -o $OUT_DIR/best_predictions

# Form an ensemble from the trained models:
echo
echo "***** RUNNING THE FUSION SCRIPT *****"
echo
python text_comprehension/eval/fusion.py -pr "$OUT_DIR/best_predictions/*.y_hat_valid" -o $OUT_DIR/best_predictions/simple_fusion.y_hat -t True --fusion_method AverageAll
