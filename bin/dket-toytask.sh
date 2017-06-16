source .py3venv/bin/activate

# define paths
TMPDIR='/tmp/dket-toytask'
DATA=$TMPDIR'/data'
DATASET=$DATA'/dataset.rio'
LOGS=$TMPDIR'/logs'

# create directories
mkdir $TMPDIR
mkdir $DATA
mkdir $LOGS

# generate the dataset
python tests/toytask/data.py --size 10000 --output $DATASET

# run the training
python dket/runtime/app.py\
    --model-name pointsoftmax\
    --batch-size 32\
    --data-files $DATASET\
    --mode train\
    --steps 1\
    --train-save-every-steps 1\
    --base-log-dir $LOGS\
    --log-level INFO\
    --log-to-stderr

# run the evaluation
# run the test

rm -rf $TMPDIR
deactivate