source .py3venv/bin/activate

# define paths
TMPDIR='/tmp/dket-toytask'
if [ -d $TMPDIR ]; then
    echo 'removing' $TMPDIR
    rm -rf $TMPDIR
fi
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
    --steps 10\
    --checkpoint-every-steps 1\
    --hparams vocabulary_size=100,shortlist_size=20,feedback_size=27\
    --base-log-dir $LOGS\
    --log-level INFO\
    --log-to-stderr

# run the evaluation
python dket/runtime/app.py\
    --model-name pointsoftmax\
    --batch-size 32\
    --data-files $DATASET\
    --mode eval\
    --epochs 1\
    --checkpoint-every-steps 3\
    --hparams vocabulary_size=100,shortlist_size=20,feedback_size=27\
    --base-log-dir $LOGS\
    --log-level INFO\
    --log-to-stderr
    --eval-check-every-sec 5\
    --eval-check-until-sec 20\
    
# run the test

# rm -rf $TMPDIR
deactivate